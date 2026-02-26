#ifndef NN_H_
#define NN_H_

#include <math.h>
#include <stdio.h>
#ifndef NN_SAFE_ALLOC
#define NN_SAFE_ALLOC 1 // Used to determine whether we want malloc with NULL check or not
#endif

#include <stdlib.h>

#ifndef NN_MALLOC
#if NN_SAFE_ALLOC
#define NN_MALLOC(size) nn_malloc_debug((size), __FILE__, __LINE__) // The safe malloc function
#else
#define NN_MALLOC malloc // Regular malloc in case where the NN_SAFE_ALLOC is turned off
#endif
#endif

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs[0])) // Used to calculate the length of arrays

#define NN_MATRIX_AT(m, i, j) (m).data[(i)*(m).stride + (j)] // m - matrix, i, j - coordinates

#define NN_INPUTS(nn) (nn).layers[0] // Get the input of the neural network
#define NN_OUTPUTS(nn) (nn).layers[(nn).layers_count - 1] // Get the output of the neural network

// Three activation function are supported in this library
typedef enum
{
    ACT_TANH,
    ACT_SIG,
    ACT_RELU
} NN_ACT;

// Matrix struct is used to represent the input of the network
typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride; // Just the size of the row
    float* data; // contiguous block is easier to re-shape
} NN_Matrix;

typedef struct 
{
    float bias;
    float act; // The value of the neuron after manipulated by the activatioin function
    size_t weights_count;
    float* weights;
} NN_Neuron;

typedef struct
{
    NN_ACT act;
    size_t neurons_count;
    NN_Neuron* neurons;
} NN_Layer;

typedef struct
{
    NN_Layer* layers;
    size_t layers_count;
} NN_Network;

void* nn_malloc_debug(size_t size, const char* file, int line);

float nn_randf(float min, float max);
float nn_sigmoidf(float x);

void nn_matrix_print(NN_Matrix mat);

NN_Neuron nn_neuron_init(size_t weights_count);
void nn_neuron_rand(NN_Neuron* neuron);

NN_Layer nn_layer_init(size_t neurons_count, NN_ACT act_func);
NN_Layer nn_layer_io_init_from_array(const float* activations, size_t activations_count);
NN_Layer* nn_layer_io_init_from_matrix(NN_Matrix mat);

NN_Network nn_network_init(const size_t* layer_sizes, size_t layers_count);
void nn_network_rand(NN_Network nn);
void nn_network_forward(NN_Network nn);
void nn_network_print(NN_Network nn);
void nn_network_set_input(NN_Network nn, NN_Layer input);
float nn_network_cost(NN_Network nn, const NN_Layer* inputs, const NN_Layer* outputs_expected, size_t entries_count);
void nn_network_finite_differences(NN_Network nn, NN_Network gradient, float epsilon, const NN_Layer* inputs, const NN_Layer* outputs_expected, size_t entries_count);
void nn_network_zero_activations(NN_Network gradient);
void nn_network_backpropagation(NN_Network nn, NN_Network gradient, const NN_Layer* inputs, const NN_Layer* outputs_expected, size_t entries_count);
void nn_network_learn(NN_Network nn, NN_Network gradient, float learning_rate);

static void __nn_network_zero(NN_Network nn);

#endif // NN_H_
#ifdef NN_IMPLEMENTATION

#if NN_SAFE_ALLOC
/*
 * This function is used whenever the Safe alloc mode is on.
 * Saves the time of checking if the allocation succeeded or not
 */
void* nn_malloc_debug(const size_t size, const char* file, const int line)
{
    void* ptr = malloc(size);
    if (ptr == NULL)
    {
        fprintf(stderr, "Error: failed to allocate memory in %s, %d\n", file, line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}
#endif

// This function generates a random real number between min and max
float nn_randf(const float min, const float max)
{
    NN_ASSERT(min < max);
    const float unit_random = (float)rand() / RAND_MAX; // Random number between 0 and 1
    const float range_size = max - min;
    return min + unit_random * range_size; // Linear interpolation
}
#define NN_RANDF() nn_randf(-1.f, 1.f)

// This function calculates the sigmoid value of x
float nn_sigmoidf(const float x)
{
    return 1.f / (1.f + expf(-x));
}

// This function receives a matrix and prints it
void nn_matrix_print(const NN_Matrix mat)
{
    printf("[\n");
    for (size_t i = 0; i < mat.rows; i++)
    {
        printf("\t");
        for (size_t j = 0; j < mat.cols; j++)
        {
            if (j == mat.cols - 1) // We go down a line
                printf("%f\n", NN_MATRIX_AT(mat, i, j));
            else
                printf("%f  ", NN_MATRIX_AT(mat, i, j));
        }
    }
    printf("]\n");
}

// This function creates a new neuron in the network
NN_Neuron nn_neuron_init(const size_t weights_count)
{
    NN_Neuron n;
    n.bias = 0.f;
    n.act = 0.f;
    n.weights_count = weights_count;
    n.weights = (float*) NN_MALLOC(sizeof(*n.weights) * n.weights_count); // The array of weights
    for (size_t i = 0; i < n.weights_count; i++)
    {
        n.weights[i] = 0.f; // Zeroing the array of weights before filling it with random numbers
    }

    return n;
}

// This function fills the neuron's bias and weights with random numbers to give the network the ability to learn them.
void nn_neuron_rand(NN_Neuron* neuron)
{
    neuron->bias = NN_RANDF();
    for (size_t i = 0; i < neuron->weights_count; i++)
    {
        neuron->weights[i] = NN_RANDF();
    }
}

// This function initializes a layer in the network
NN_Layer nn_layer_init(const size_t neurons_count, const NN_ACT act_func)
{
    NN_Neuron* neurons = (NN_Neuron*) NN_MALLOC(sizeof(NN_Neuron)*neurons_count); // The array of neurons in the layer
    return (NN_Layer)
    {
        .act = act_func, // The activation function is decided on the layer level
        .neurons_count = neurons_count,
        .neurons = neurons
    };
}

// This function creates the first layer of the network. It only represents the input
NN_Layer nn_layer_io_init_from_array(const float* activations, const size_t activations_count)
{
    NN_Neuron* neurons = (NN_Neuron*) NN_MALLOC(sizeof(NN_Neuron)*activations_count);

    // Creating the layer, without activation function since it doesn't do any computations
    const NN_Layer layer =
    {
        .neurons_count = activations_count,
        .neurons = neurons
    };

    // Creating the neurons of this layer
    for (size_t i = 0; i < activations_count; i++)
    {
        layer.neurons[i] = (NN_Neuron)
        {
            .act = activations[i],
            .weights_count = 0,
            .weights = NULL,
            .bias = 0.f
        };
    }

    return layer;
}

// This function creates an array of 'first layers' for each input in the matrix
NN_Layer* nn_layer_io_init_from_matrix(const NN_Matrix mat)
{
    NN_Layer* layers = (NN_Layer*) NN_MALLOC(sizeof(NN_Layer)*mat.rows); // Allocating the layers

    for (size_t i = 0; i < mat.rows; i++)
    {
        NN_Neuron* neurons = (NN_Neuron*) NN_MALLOC(sizeof(NN_Neuron)*mat.cols); // Allocating the array of neurons
        layers[i] = (NN_Layer)
        {
            .neurons = neurons,
            .neurons_count = mat.cols
        }; // Creating the first layer for the i'th input

        // Creating the neurons in the layer
        for (size_t j = 0; j < mat.cols; j++)
        {
            layers[i].neurons[j] = (NN_Neuron)
            {
                .act = NN_MATRIX_AT(mat, i, j),
                .weights_count = 0,
                .weights = NULL,
                .bias = 0.f
            };
        }
        
    }

    return layers;
}

// This function initializes an entire neural network
NN_Network nn_network_init(const size_t* layer_sizes, const size_t layers_count)
{
    NN_Network nn;

    nn.layers_count = layers_count;
    nn.layers = (NN_Layer*) NN_MALLOC((layers_count) * sizeof(NN_Layer));

    // Initializing the layers
    for (size_t i = 0; i < nn.layers_count; ++i)
    {
        NN_Layer* l = &nn.layers[i];
        l->act = ACT_SIG; // For now only supporting sigmoid
        l->neurons_count = layer_sizes[i];
        l->neurons = (NN_Neuron*) NN_MALLOC(l->neurons_count * sizeof(NN_Neuron));

        /*
         * The number of weights there are in the first level is 0 and in any other level is the number of neurons in
         * the previous layer
         */
        const size_t neuron_weights_count = (i == 0)
            ? 0
            : nn.layers[i - 1].neurons_count;

        // Initializing each neuron
        for (size_t j = 0; j < l->neurons_count; j++)
        {
            l->neurons[j] = nn_neuron_init(neuron_weights_count);
        }
    }

    return nn;
}

// This function generates random 'starting points' for the neurons weights and biases for them to be able to learn.
void nn_network_rand(const NN_Network nn)
{
    /* Starting from i = 1 because the first layer (which represents the input layer) doesn't have any weights at all,
     * and we need the bias of the neurons in that layer to stay 0
     */
    for (size_t i = 1; i < nn.layers_count; ++i) // Iterating over the layers
    {
        const NN_Layer* l = &nn.layers[i]; // Easy access to the layer

        // For each neuron in the layer we generate weights and biases using the nn_neuron_rand function
        for (size_t j = 0; j < l->neurons_count; j++)
        {
            nn_neuron_rand(&l->neurons[j]);
        }
    }
}

// This function sets the first layer in the network to be the input
void nn_network_set_input(const NN_Network nn, const NN_Layer input)
{
    const NN_Layer input_layer = nn.layers[0]; // Easy access to first layer

    // Asserting the number of neurons in the first layer equals to the number of input entries
    NN_ASSERT(input_layer.neurons_count == input.neurons_count);

    // Assigning the input to every neuron in the layer
    for (size_t i = 0; i < input_layer.neurons_count; i++)
    {
        NN_Neuron* neuron = &input_layer.neurons[i]; // Easy access to neuron

        // Assigning the input value to the neuron's 'act' parameter - not really passing through any activation func
        neuron->act = input.neurons[i].act;
    }
}

// This function performs a forward propagation - calculates the value of the nodes in the network
void nn_network_forward(const NN_Network nn)
{

    // Zeroing the network except for the first layer of the inputs to avoid errors
    __nn_network_zero(nn);

    /*
     * Iterating over all the layers (except the first one of the inputs) and calculating the value of each neuron
     * in the network
     */
    for (size_t i = 1; i < nn.layers_count; i++)
    {

        // Easy access
        const NN_Layer* layer = &nn.layers[i];
        const NN_Layer* layer_prev = &nn.layers[i - 1];

        // Iterating over all the neurons in the layer
        for (size_t j = 0; j < layer->neurons_count; j++)
        {
            NN_Neuron* neuron = &layer->neurons[j]; // Easy access
            float sum = neuron->bias; // sum will be the new value of the neuron (before the activation func if needed)

            /* Iterating over all the neurons from the previous layer, multiplying their 'act' by the weight our
             * neuron have in relation to this previous neuron - can be 0
             */
            for (size_t k = 0; k < neuron->weights_count; ++k)
            {
                const NN_Neuron* neuron_prev = &layer_prev->neurons[k];
                sum += neuron_prev->act * neuron->weights[k];
            }

            // Last layer does not go through the activation function
            neuron->act = (i == nn.layers_count - 1)
                ? sum
                : nn_sigmoidf(sum);
        }
    }
}

/*
 * This function is used to zero all the entries in the neural network (except first layer) in order to avoid errors
 * before propagating.
 * This is a static function because it is only a helper - we do not allow users to call it
 */
static void __nn_network_zero(const NN_Network nn)
{

    // Starting from i = 1 because we only zero the non-input layers.
    for (size_t i = 1; i < nn.layers_count; i++)
    {
        const NN_Layer* layer = &nn.layers[i]; // Easy access
        for (size_t j = 0; j < layer->neurons_count; j++)
        {
            NN_Neuron* neuron = &layer->neurons[j]; // Easy access
            neuron->act = 0.f; // Zero the act of the neuron
        }
    }
}

// This function is used to print the entire neural network
void nn_network_print(const NN_Network nn)
{
    printf("Neural Network:\n");

    // Iterating over the layers
    for (size_t i = 0; i < nn.layers_count; i++)
    {
        printf("Layer%lu {\n", i);
        const NN_Layer* layer = &nn.layers[i];

        // Iterating over the neurons in the current layer
        for (size_t j = 0; j < layer->neurons_count; j++)
        {
            const NN_Neuron* neuron = &layer->neurons[j];
            printf("\tNeuron%lu {\n", j);
            printf("\t\tact = %f,\n", neuron->act);
            printf("\t\tbias = %f,\n", neuron->bias);
            printf("\t\tweights = [\n");

            // Iterating over the weights of the neuron in the current layer
            for (size_t k = 0; k < neuron->weights_count; k++)
            {

                // Distinguish between the last weight and other weights because of the comma.
                if (k == neuron->weights_count - 1)
                    printf("\t\t\t weight%lu = %f\n", k, neuron->weights[k]);
                else
                    printf("\t\t\t weight%lu = %f,\n", k, neuron->weights[k]);
            }
            printf("\t\t]\n");
        }
        printf("}\n");
    }
}

// Evaluate the cost function of the network
float nn_network_cost(const NN_Network nn, const NN_Layer* inputs, const NN_Layer* outputs_expected, const size_t entries_count)
{
    float cost = 0.f; // We start with cost = 0 and add to it as needed

    // Iterating over all the different inputs
    for (size_t i = 0; i < entries_count; i++)
    {
        float partial_cost = 0.f; // The square errors per input

        const NN_Layer* input = &inputs[i];
        const NN_Layer* output = &outputs_expected[i];

        /*
         * Asserting the number of neurons in the first layer of the network equals to the number of neurons in the
         * input, and doing the same for the output.
         */
        NN_ASSERT(NN_INPUTS(nn).neurons_count == input->neurons_count);
        NN_ASSERT(NN_OUTPUTS(nn).neurons_count == output->neurons_count);

        // Inserting the input to the network and propagating it forward!
        nn_network_set_input(nn, *input);
        nn_network_forward(nn);

        // Iterating over the neurons in the output to see how far it is from the expected output
        for (size_t j = 0; j < output->neurons_count; j++)
        {
            const float prediction = NN_OUTPUTS(nn).neurons[j].act; // What out network said
            const float expected   = output->neurons[j].act; // What we expected
            const float distance   = prediction - expected;
            partial_cost     += distance * distance; // Adding the distance squared
        }
        cost += partial_cost; // Adding to the total error
    }

    return cost / (float)entries_count; // taking the Mean squared error
}

/*
 * This function fills a new neural network which is called gradient. The entries of this network form the gradient of
 * the cost function.
 * For example, in the n'th neuron in the m'th level, the value of its k'th weight represents the change in the cost
 * function. It is the partial derivative of the cost function with respect to the variable which is the k'th weight
 * of the n'th neuron in the m'th layer (lets denote it by v for now).
 * The partial derivative is estimated by: (cost(...v + epsilon...) - cost(...v...)) / epsilon)
 * (We do the same for the biases too)
 */
void nn_network_finite_differences(const NN_Network nn, const NN_Network gradient, const float epsilon,
                                 const NN_Layer* inputs, const NN_Layer* outputs_expected, const size_t entries_count)
{
    const float cost_original = nn_network_cost(nn, inputs, outputs_expected, entries_count); // The original cost


    for (size_t i = 1; i < nn.layers_count; i++)
    {
        const NN_Layer* l = &nn.layers[i];
        for (size_t j = 0; j < l->neurons_count; j++)
        {
            NN_Neuron* neuron = &l->neurons[j];
            float temp;
            float cost_new;
            float partial_derivative;

            for (size_t k = 0; k < neuron->weights_count; k++)
            {
                temp = neuron->weights[k]; // Remembering the previous value
                neuron->weights[k] += epsilon; // Applying the new value to the network

                // The new cost is obtained now by calculating the cost of the changed network
                cost_new = nn_network_cost(nn, inputs, outputs_expected, entries_count);

                // Calculating the partial derivative using the formula written earlier
                partial_derivative = (cost_new - cost_original) / epsilon;

                // Filling the gradient network
                gradient.layers[i].neurons[j].weights[k] = partial_derivative;

                // Back to the original value
                neuron->weights[k] = temp;
            }

            // Now doing the same for the biases

            temp = neuron->bias;
            neuron->bias += epsilon;
            cost_new = nn_network_cost(nn, inputs, outputs_expected, entries_count);
            partial_derivative = (cost_new - cost_original) / epsilon;
            gradient.layers[i].neurons[j].bias = partial_derivative;
            neuron->bias = temp;
        }
    }
}

// This function zeros the gradient network (used to avoid errors between different inputs)
void nn_network_zero_activations(const NN_Network gradient)
{
    for (size_t l = 0; l < gradient.layers_count; l++)
    {
        for (size_t m = 0; m < gradient.layers[l].neurons_count; ++m)
            gradient.layers[l].neurons[m].act = 0;
    }
}

void nn_network_backpropagation(const NN_Network nn, const NN_Network gradient, const NN_Layer* inputs,
    const NN_Layer* outputs_expected, const size_t entries_count)
{

    // Asserting the number of layers in the gradient equals to the number of layers in the original network
    NN_ASSERT(nn.layers_count == gradient.layers_count);

    const NN_Layer* layer_last = &NN_OUTPUTS(nn); // Easy access to the last layer in our original network

    // Iterating over the inputs
    for (size_t i = 0; i < entries_count; i++)
    {

        // Inserting the input to the network and propagating it forward
        nn_network_set_input(nn, inputs[i]);
        nn_network_forward(nn);

        // Now iterating over the neurons in the last layer! We want to see how much our neurons affects it
        for (size_t j = 0; j < layer_last->neurons_count; j++)
        {
            // Initialising dynamic programming
            nn_network_zero_activations(gradient);

            /*
             * The activation of every neuron in the gradient network represents how much it affects the last
             * layer. The partial derivative of the neurons in the last level with respect to themselves is
             * of course 1.
             * This is the base of our dynamic programming
             */
            for (size_t d = 0; d < layer_last->neurons_count; d++)
            {
                gradient.layers[gradient.layers_count - 1].neurons[d].act = (d == j)
                    ? 1.f
                    : 0.f;
            }

            const float prediction = layer_last->neurons[j].act;
            const float expected = outputs_expected[i].neurons[j].act;

            // l starts at nn.layers_count -1, weird line. Did not want to deal with size_t and int comparisons
            for (size_t l = nn.layers_count; l-- > 0;)
            {
                const NN_Layer* layer = &nn.layers[l]; // current layer

                /*
                 * Iterating over the neurons in the current layer. Our goal is to find how much the weights and biases
                 * need to change
                 */
                for (size_t m = 0; m < layer->neurons_count; m++)
                {
                    const NN_Neuron* neuron = &layer->neurons[m]; // The current neuron

                    // The derivative of the prediction j with respect to the activation of neuron
                    float d_pred_j_d_act = 0.f;

                    // Iterating over the neurons in the NEXT layer (after neuron) in order to use chain rule
                    for (size_t k = 0; k < nn.layers[l+1].neurons_count; k++)
                    {
                        // The derivative of prediction j with respect to activation of neuron in next layer
                        const float d_pred_j_d_act_next = gradient.layers[l+1].neurons[k].act;

                        // The derivative of neuron in next layer with respect to 'neuron'
                        const float d_act_next_d_act = nn.layers[l+1].neurons[k].act *
                            (1 - nn.layers[l+1].neurons[k].act) * nn.layers[l+1].neurons[k].weights[m];

                        // Chain rule
                        d_pred_j_d_act += d_pred_j_d_act_next * d_act_next_d_act;
                    }

                    // Saving the result in the gradient network
                    gradient.layers[l].neurons[m].act = d_pred_j_d_act;


                    /*
                     * Iterating over the weights of our current neuron in order to check how much it affects the
                     * neuron j in the last layer
                     */
                    for (size_t t = 0; t < neuron->weights_count; t++)
                    {

                        // Derivative of prediction j with respect to weight t of neuron.
                        float d_pred_j_d_weight = 0.f;

                        // If neuron is from the last layer
                        if (l == nn.layers_count - 1)
                        {

                            /*
                             * If neuron = prediction, the derivative is exactly the t'th neuron in the previous layer
                             * otherwise, it is 0.
                             */
                            d_pred_j_d_weight = (m == j)
                                ? nn.layers[l-1].neurons[t].act
                                : 0.f;

                        }

                        // If neuron is not from the last layer
                        else
                        {
                            // The derivative of the activation of neuron with respect to the weight
                            const float d_act_d_weight = nn.layers[l].neurons[m].act
                                * (1 - nn.layers[l].neurons[m].act) * nn.layers[l-1].neurons[t].act;

                            // Using chain rule to calculate the affect the weight has on the prediction
                            d_pred_j_d_weight = d_pred_j_d_act * d_act_d_weight;
                        }

                        /*
                         * Updating the affect the t weight of m neuron in l level have on the cost function
                         * Using the derivative of the cost function
                         */
                        gradient.layers[l].neurons[m].weights[t] += 2 * (prediction - expected) * d_pred_j_d_weight;
                    }

                    // Calculating the derivative of the bias
                    float d_pred_j_d_bias = 0.f;

                    // If neuron is from the last layer
                    if (l == nn.layers_count - 1)
                    {

                        //If neuron = prediction, the derivative is exactly 1, otherwise, 0
                        d_pred_j_d_bias = (m == j)
                            ? 1.f
                            : 0.f;
                    }
                    else
                    {
                        /*
                         * MASSIVE OPTIMIZATION SHORTCUT
                         * Originally, we had an entire inner loop here iterating over the NEXT layer
                         * to calculate how this neuron's bias affects the final prediction.
                         * BUT, we can skip that loop entirely!
                         * * Math breakdown using the Chain Rule:
                         * d_pred_j_d_bias = (d_pred_j_d_act) * (d_act_d_bias)
                         * * 1. The first part (d_pred_j_d_act) represents the sum of all influences
                         * this neuron has on the next layer.
                         * we ALREADY calculated this exact sum a few lines above!
                         * We did it for the weights and stored it in the variable `d_pred_j_d_act`.
                         * * 2. The second part (d_act_d_bias) asks: how much does the bias affect
                         * this neuron's OWN activation?
                         * Since the activation is a sigmoid function: act = sigmoid(weights*inputs + bias),
                         * the derivative of the activation with respect to the bias is simply the
                         * derivative of the sigmoid function itself: act * (1 - act).
                         * * By multiplying these two parts together, we recycle the heavy mathematical
                         * lifting we already did. We get the exact same result in O(1) instead of O(N),
                         * cutting the bias calculation time to a fraction!
                         */
                        d_pred_j_d_bias = d_pred_j_d_act * neuron->act * (1 - neuron->act);
                    }

                    /*
                     * Updating the affect the bias of m neuron in l level have on the cost function
                     * Using the derivative of the cost function
                     */
                    gradient.layers[l].neurons[m].bias += 2 * (prediction - expected) * d_pred_j_d_bias;
                }
            }
        }
    }

    // Iterating over the gradient network and normalizing the result
    for (size_t l = 0; l < gradient.layers_count; l++)
    {
        for (size_t m = 0; m < gradient.layers[l].neurons_count; m++)
        {

            // Dividing by the number of inputs in order for it to really be "Mean Squared Error"
            gradient.layers[l].neurons[m].bias /= (float)entries_count;

            // Now doing the same for every weight
            for (size_t t = 0; t < gradient.layers[l].neurons[m].weights_count; t++)
            {
                gradient.layers[l].neurons[m].weights[t] /= (float)entries_count;
            }
        }
    }
}

void nn_network_learn(NN_Network nn, NN_Network gradient, float learning_rate)
{
    NN_ASSERT(nn.layers_count == gradient.layers_count);
    for (size_t i = 1; i < nn.layers_count; ++i)
    {
        NN_Layer* l = &nn.layers[i];
        NN_ASSERT(l->neurons_count == gradient.layers[i].neurons_count);
        for (size_t j = 0; j < l->neurons_count; ++j)
        {
            NN_Neuron* neuron = &l->neurons[j];
            for (size_t k = 0; k < neuron->weights_count; ++k)
            {
                neuron->weights[k] -= gradient.layers[i].neurons[j].weights[k] * learning_rate;
            }
            neuron->bias -= gradient.layers[i].neurons[j].bias * learning_rate;
        }
    }
}
#endif // NN_IMPLEMNTATION
