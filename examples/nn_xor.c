#define NN_IMPLEMENTATION
#include "../nn.h"

float training_data[][3] = {
    { 0.f, 0.f, 0.f },
    { 1.f, 0.f, 1.f },
    { 0.f, 1.f, 1.f },
    { 1.f, 1.f, 0.f },
};

int main(void)
{
    srand(69);
    size_t layer_sizes[] = { 2, 2, 1 };
    NN_Network nn = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    NN_Network gradient = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    nn_network_rand(nn);

    NN_Matrix in =
    {
        .rows = 4,
        .cols = 2,
        .stride = 3,
        .data = &training_data[0][0]
    };
    NN_Matrix out =
    {
        .rows = 4,
        .cols = 1,
        .stride = 3,
        .data = &training_data[0][2]
    };

    NN_Layer* inputs = nn_layer_io_init_from_matrix(in);
    NN_Layer* outputs = nn_layer_io_init_from_matrix(out);

    for (size_t epoch = 0; epoch < 10 * 1000; ++epoch)
    {
        if (epoch % 1000 == 0)
            printf("epoch: %lu, cost: %f\n", epoch, nn_network_cost(nn, inputs, outputs, in.rows));
        nn_network_backpropagation(nn, gradient, inputs, outputs, in.rows);
        nn_network_learn(nn, gradient, 1e-1);
    }

    float cost = nn_network_cost(nn, inputs, outputs, in.rows);
    printf("Finished training. Cost = %f\n", cost);
    printf("Result network: \n");
    nn_network_print(nn);
    printf("Testing the model: \n");

    for (size_t i = 0; i < in.rows; ++i)
    {
        nn_network_set_input(nn, inputs[i]);
        nn_network_forward(nn);
        printf("%f ^ %f = %f\n", inputs[i].neurons[0].act, inputs[i].neurons[1].act, NN_OUTPUTS(nn).neurons[0].act);
    }

    return 0;
}
