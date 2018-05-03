[
    {
        "name": "Function_0",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_25",
                "op": "Parameter",
                "outputs": [
                    "Parameter_25_0"
                ],
                "shape": [
                    50
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_15",
                "op": "Parameter",
                "outputs": [
                    "Parameter_15_0"
                ],
                "shape": [
                    1
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_5",
                "op": "Parameter",
                "outputs": [
                    "Parameter_5_0"
                ],
                "shape": [
                    1,
                    50
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1_0"
                ],
                "shape": [
                    1,
                    1
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_0",
                "op": "Parameter",
                "outputs": [
                    "Parameter_0_0"
                ],
                "shape": [
                    50,
                    1
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_35",
                "op": "Parameter",
                "outputs": [
                    "Parameter_35_0"
                ],
                "shape": [
                    2,
                    1
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_6",
                "op": "Parameter",
                "outputs": [
                    "Parameter_6_0"
                ],
                "shape": [
                    2,
                    1
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_2",
                "op": "Constant",
                "outputs": [
                    "Constant_2_0"
                ],
                "shape": [],
                "value": [
                    "0"
                ]
            },
            {
                "axes": [
                    1,
                    2
                ],
                "inputs": [
                    "Parameter_25"
                ],
                "name": "Broadcast_26",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_26_0"
                ],
                "shape": [
                    50,
                    1,
                    2
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Parameter_15"
                ],
                "name": "Broadcast_16",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_16_0"
                ],
                "shape": [
                    1,
                    2
                ]
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Parameter_5"
                ],
                "name": "Reshape_8",
                "op": "Reshape",
                "output_shape": [
                    1,
                    50
                ],
                "outputs": [
                    "Reshape_8_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Parameter_0"
                ],
                "name": "Reshape_21",
                "op": "Reshape",
                "output_shape": [
                    50,
                    1
                ],
                "outputs": [
                    "Reshape_21_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_6"
                ],
                "name": "OneHot_7",
                "one_hot_axis": 0,
                "op": "OneHot",
                "outputs": [
                    "OneHot_7_0"
                ],
                "shape": [
                    50,
                    2,
                    1
                ]
            },
            {
                "axes": [
                    0,
                    1
                ],
                "inputs": [
                    "Constant_2"
                ],
                "name": "Broadcast_3",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_3_0"
                ],
                "shape": [
                    1,
                    2
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "OneHot_7"
                ],
                "name": "Reshape_9",
                "op": "Reshape",
                "output_shape": [
                    50,
                    2
                ],
                "outputs": [
                    "Reshape_9_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Broadcast_3"
                ],
                "name": "Dot_4",
                "op": "Dot",
                "outputs": [
                    "Dot_4_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Reshape_8",
                    "Reshape_9"
                ],
                "name": "Dot_10",
                "op": "Dot",
                "outputs": [
                    "Dot_10_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Dot_10"
                ],
                "name": "Reshape_11",
                "op": "Reshape",
                "output_shape": [
                    1,
                    2,
                    1
                ],
                "outputs": [
                    "Reshape_11_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_11"
                ],
                "lower_bounds": [
                    0,
                    0,
                    0
                ],
                "name": "Slice_12",
                "op": "Slice",
                "outputs": [
                    "Slice_12_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    1,
                    2,
                    1
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_12"
                ],
                "name": "Reshape_13",
                "op": "Reshape",
                "output_shape": [
                    1,
                    2
                ],
                "outputs": [
                    "Reshape_13_0"
                ]
            },
            {
                "inputs": [
                    "Dot_4",
                    "Reshape_13"
                ],
                "name": "Add_14",
                "op": "Add",
                "outputs": [
                    "Add_14_0"
                ]
            },
            {
                "inputs": [
                    "Add_14",
                    "Broadcast_16"
                ],
                "name": "Add_17",
                "op": "Add",
                "outputs": [
                    "Add_17_0"
                ]
            },
            {
                "inputs": [
                    "Add_17"
                ],
                "name": "Tanh_18",
                "op": "Tanh",
                "outputs": [
                    "Tanh_18_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Tanh_18"
                ],
                "name": "Broadcast_19",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_19_0"
                ],
                "shape": [
                    1,
                    1,
                    2
                ]
            },
            {
                "axis": 1,
                "inputs": [
                    "Broadcast_19"
                ],
                "name": "Concat_20",
                "op": "Concat",
                "outputs": [
                    "Concat_20_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Concat_20"
                ],
                "name": "Reshape_22",
                "op": "Reshape",
                "output_shape": [
                    1,
                    2
                ],
                "outputs": [
                    "Reshape_22_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_21",
                    "Reshape_22"
                ],
                "name": "Dot_23",
                "op": "Dot",
                "outputs": [
                    "Dot_23_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Dot_23"
                ],
                "name": "Reshape_24",
                "op": "Reshape",
                "output_shape": [
                    50,
                    1,
                    2
                ],
                "outputs": [
                    "Reshape_24_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_24",
                    "Broadcast_26"
                ],
                "name": "Add_27",
                "op": "Add",
                "outputs": [
                    "Add_27_0"
                ]
            },
            {
                "inputs": [
                    "Add_27"
                ],
                "name": "Max_28",
                "op": "Max",
                "outputs": [
                    "Max_28_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Max_28"
                ],
                "name": "Broadcast_29",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_29_0"
                ],
                "shape": [
                    50,
                    1,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_27",
                    "Broadcast_29"
                ],
                "name": "Subtract_30",
                "op": "Subtract",
                "outputs": [
                    "Subtract_30_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_30"
                ],
                "name": "Exp_31",
                "op": "Exp",
                "outputs": [
                    "Exp_31_0"
                ]
            },
            {
                "inputs": [
                    "Exp_31"
                ],
                "name": "Sum_32",
                "op": "Sum",
                "outputs": [
                    "Sum_32_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Sum_32"
                ],
                "name": "Broadcast_33",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_33_0"
                ],
                "shape": [
                    50,
                    1,
                    2
                ]
            },
            {
                "inputs": [
                    "Exp_31",
                    "Broadcast_33"
                ],
                "name": "Divide_34",
                "op": "Divide",
                "outputs": [
                    "Divide_34_0"
                ]
            },
            {
                "inputs": [
                    "Divide_34"
                ],
                "name": "Result_36",
                "op": "Result",
                "outputs": [
                    "Result_36_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_6",
            "Parameter_35",
            "Parameter_0",
            "Parameter_1",
            "Parameter_5",
            "Parameter_15",
            "Parameter_25"
        ],
        "result": [
            "Result_36"
        ]
    }
]
