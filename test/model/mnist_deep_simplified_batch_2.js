[
    {
        "name": "Function_21",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_2819",
                "op": "Parameter",
                "outputs": [
                    "Parameter_2819_0"
                ],
                "shape": [
                    2,
                    784
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_2818",
                "op": "Parameter",
                "outputs": [
                    "Parameter_2818_0"
                ],
                "shape": [
                    10
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_2817",
                "op": "Parameter",
                "outputs": [
                    "Parameter_2817_0"
                ],
                "shape": [
                    100,
                    10
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_2816",
                "op": "Parameter",
                "outputs": [
                    "Parameter_2816_0"
                ],
                "shape": [
                    100
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_2815",
                "op": "Parameter",
                "outputs": [
                    "Parameter_2815_0"
                ],
                "shape": [
                    539,
                    100
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_2814",
                "op": "Parameter",
                "outputs": [
                    "Parameter_2814_0"
                ],
                "shape": [
                    11
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_2813",
                "op": "Parameter",
                "outputs": [
                    "Parameter_2813_0"
                ],
                "shape": [
                    5,
                    5,
                    5,
                    11
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_2812",
                "op": "Parameter",
                "outputs": [
                    "Parameter_2812_0"
                ],
                "shape": [
                    5
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_2811",
                "op": "Parameter",
                "outputs": [
                    "Parameter_2811_0"
                ],
                "shape": [
                    5,
                    5,
                    1,
                    5
                ]
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Parameter_2819"
                ],
                "name": "Reshape_2820",
                "op": "Reshape",
                "output_shape": [
                    2,
                    28,
                    28,
                    1
                ],
                "outputs": [
                    "Reshape_2820_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Parameter_2818"
                ],
                "name": "Broadcast_2846",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_2846_0"
                ],
                "shape": [
                    2,
                    10
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Parameter_2816"
                ],
                "name": "Broadcast_2842",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_2842_0"
                ],
                "shape": [
                    2,
                    100
                ]
            },
            {
                "axes": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Parameter_2814"
                ],
                "name": "Broadcast_2835",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_2835_0"
                ],
                "shape": [
                    2,
                    7,
                    7,
                    11
                ]
            },
            {
                "input_order": [
                    3,
                    2,
                    0,
                    1
                ],
                "inputs": [
                    "Parameter_2813"
                ],
                "name": "Reshape_2832",
                "op": "Reshape",
                "output_shape": [
                    11,
                    5,
                    5,
                    5
                ],
                "outputs": [
                    "Reshape_2832_0"
                ]
            },
            {
                "axes": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Parameter_2812"
                ],
                "name": "Broadcast_2825",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_2825_0"
                ],
                "shape": [
                    2,
                    14,
                    14,
                    5
                ]
            },
            {
                "input_order": [
                    3,
                    2,
                    0,
                    1
                ],
                "inputs": [
                    "Parameter_2811"
                ],
                "name": "Reshape_2822",
                "op": "Reshape",
                "output_shape": [
                    5,
                    1,
                    5,
                    5
                ],
                "outputs": [
                    "Reshape_2822_0"
                ]
            },
            {
                "input_order": [
                    0,
                    3,
                    1,
                    2
                ],
                "inputs": [
                    "Reshape_2820"
                ],
                "name": "Reshape_2821",
                "op": "Reshape",
                "output_shape": [
                    2,
                    1,
                    28,
                    28
                ],
                "outputs": [
                    "Reshape_2821_0"
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_2821",
                    "Reshape_2822"
                ],
                "name": "Convolution_2823",
                "op": "Convolution",
                "outputs": [
                    "Convolution_2823_0"
                ],
                "padding_above": [
                    2,
                    2
                ],
                "padding_below": [
                    1,
                    1
                ],
                "window_dilation_strides": [
                    1,
                    1
                ],
                "window_movement_strides": [
                    2,
                    2
                ]
            },
            {
                "input_order": [
                    0,
                    2,
                    3,
                    1
                ],
                "inputs": [
                    "Convolution_2823"
                ],
                "name": "Reshape_2824",
                "op": "Reshape",
                "output_shape": [
                    2,
                    14,
                    14,
                    5
                ],
                "outputs": [
                    "Reshape_2824_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_2824",
                    "Broadcast_2825"
                ],
                "name": "Add_2826",
                "op": "Add",
                "outputs": [
                    "Add_2826_0"
                ]
            },
            {
                "inputs": [
                    "Add_2826",
                    "Add_2826"
                ],
                "name": "Multiply_2827",
                "op": "Multiply",
                "outputs": [
                    "Multiply_2827_0"
                ]
            },
            {
                "input_order": [
                    0,
                    3,
                    1,
                    2
                ],
                "inputs": [
                    "Multiply_2827"
                ],
                "name": "Reshape_2828",
                "op": "Reshape",
                "output_shape": [
                    2,
                    5,
                    14,
                    14
                ],
                "outputs": [
                    "Reshape_2828_0"
                ]
            },
            {
                "include_padding_in_avg_computation": false,
                "inputs": [
                    "Reshape_2828"
                ],
                "name": "AvgPool_2829",
                "op": "AvgPool",
                "outputs": [
                    "AvgPool_2829_0"
                ],
                "padding_above": [
                    1,
                    1
                ],
                "padding_below": [
                    1,
                    1
                ],
                "window_movement_strides": [
                    1,
                    1
                ],
                "window_shape": [
                    3,
                    3
                ]
            },
            {
                "input_order": [
                    0,
                    2,
                    3,
                    1
                ],
                "inputs": [
                    "AvgPool_2829"
                ],
                "name": "Reshape_2830",
                "op": "Reshape",
                "output_shape": [
                    2,
                    14,
                    14,
                    5
                ],
                "outputs": [
                    "Reshape_2830_0"
                ]
            },
            {
                "input_order": [
                    0,
                    3,
                    1,
                    2
                ],
                "inputs": [
                    "Reshape_2830"
                ],
                "name": "Reshape_2831",
                "op": "Reshape",
                "output_shape": [
                    2,
                    5,
                    14,
                    14
                ],
                "outputs": [
                    "Reshape_2831_0"
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_2831",
                    "Reshape_2832"
                ],
                "name": "Convolution_2833",
                "op": "Convolution",
                "outputs": [
                    "Convolution_2833_0"
                ],
                "padding_above": [
                    2,
                    2
                ],
                "padding_below": [
                    1,
                    1
                ],
                "window_dilation_strides": [
                    1,
                    1
                ],
                "window_movement_strides": [
                    2,
                    2
                ]
            },
            {
                "input_order": [
                    0,
                    2,
                    3,
                    1
                ],
                "inputs": [
                    "Convolution_2833"
                ],
                "name": "Reshape_2834",
                "op": "Reshape",
                "output_shape": [
                    2,
                    7,
                    7,
                    11
                ],
                "outputs": [
                    "Reshape_2834_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_2834",
                    "Broadcast_2835"
                ],
                "name": "Add_2836",
                "op": "Add",
                "outputs": [
                    "Add_2836_0"
                ]
            },
            {
                "input_order": [
                    0,
                    3,
                    1,
                    2
                ],
                "inputs": [
                    "Add_2836"
                ],
                "name": "Reshape_2837",
                "op": "Reshape",
                "output_shape": [
                    2,
                    11,
                    7,
                    7
                ],
                "outputs": [
                    "Reshape_2837_0"
                ]
            },
            {
                "include_padding_in_avg_computation": false,
                "inputs": [
                    "Reshape_2837"
                ],
                "name": "AvgPool_2838",
                "op": "AvgPool",
                "outputs": [
                    "AvgPool_2838_0"
                ],
                "padding_above": [
                    1,
                    1
                ],
                "padding_below": [
                    1,
                    1
                ],
                "window_movement_strides": [
                    1,
                    1
                ],
                "window_shape": [
                    3,
                    3
                ]
            },
            {
                "input_order": [
                    0,
                    2,
                    3,
                    1
                ],
                "inputs": [
                    "AvgPool_2838"
                ],
                "name": "Reshape_2839",
                "op": "Reshape",
                "output_shape": [
                    2,
                    7,
                    7,
                    11
                ],
                "outputs": [
                    "Reshape_2839_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2,
                    3
                ],
                "inputs": [
                    "Reshape_2839"
                ],
                "name": "Reshape_2840",
                "op": "Reshape",
                "output_shape": [
                    2,
                    539
                ],
                "outputs": [
                    "Reshape_2840_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_2840",
                    "Parameter_2815"
                ],
                "name": "Dot_2841",
                "op": "Dot",
                "outputs": [
                    "Dot_2841_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_2841",
                    "Broadcast_2842"
                ],
                "name": "Add_2843",
                "op": "Add",
                "outputs": [
                    "Add_2843_0"
                ]
            },
            {
                "inputs": [
                    "Add_2843",
                    "Add_2843"
                ],
                "name": "Multiply_2844",
                "op": "Multiply",
                "outputs": [
                    "Multiply_2844_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_2844",
                    "Parameter_2817"
                ],
                "name": "Dot_2845",
                "op": "Dot",
                "outputs": [
                    "Dot_2845_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_2845",
                    "Broadcast_2846"
                ],
                "name": "Add_2847",
                "op": "Add",
                "outputs": [
                    "Add_2847_0"
                ]
            },
            {
                "inputs": [
                    "Add_2847"
                ],
                "name": "Result_2850",
                "op": "Result",
                "outputs": [
                    "Result_2850_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_2811",
            "Parameter_2812",
            "Parameter_2813",
            "Parameter_2814",
            "Parameter_2815",
            "Parameter_2816",
            "Parameter_2817",
            "Parameter_2818",
            "Parameter_2819"
        ],
        "result": [
            "Result_2850"
        ]
    }
]