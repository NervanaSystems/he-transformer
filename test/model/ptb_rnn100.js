[
    {
        "name": "Function_0",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_64",
                "op": "Parameter",
                "outputs": [
                    "Parameter_64_0"
                ],
                "shape": [
                    50
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_9",
                "op": "Parameter",
                "outputs": [
                    "Parameter_9_0"
                ],
                "shape": [
                    100,
                    50
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
                    100
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
                    100,
                    100
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
                    100
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_67",
                "op": "Parameter",
                "outputs": [
                    "Parameter_67_0"
                ],
                "shape": [
                    2,
                    4
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_10",
                "op": "Parameter",
                "outputs": [
                    "Parameter_10_0"
                ],
                "shape": [
                    2,
                    4
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_7",
                "op": "Constant",
                "outputs": [
                    "Constant_7_0"
                ],
                "shape": [],
                "value": [
                    "0.01"
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
                    "Parameter_64"
                ],
                "name": "Broadcast_65",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_65_0"
                ],
                "shape": [
                    50,
                    4,
                    2
                ]
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Parameter_9"
                ],
                "name": "Reshape_12",
                "op": "Reshape",
                "output_shape": [
                    100,
                    50
                ],
                "outputs": [
                    "Reshape_12_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Parameter_5"
                ],
                "name": "Broadcast_6",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_6_0"
                ],
                "shape": [
                    100,
                    2
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
                "name": "Reshape_60",
                "op": "Reshape",
                "output_shape": [
                    50,
                    100
                ],
                "outputs": [
                    "Reshape_60_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_10"
                ],
                "name": "OneHot_11",
                "one_hot_axis": 0,
                "op": "OneHot",
                "outputs": [
                    "OneHot_11_0"
                ],
                "shape": [
                    50,
                    2,
                    4
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Constant_7"
                ],
                "name": "Broadcast_8",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_8_0"
                ],
                "shape": [
                    2
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
                    100,
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
                    "OneHot_11"
                ],
                "name": "Reshape_13",
                "op": "Reshape",
                "output_shape": [
                    50,
                    8
                ],
                "outputs": [
                    "Reshape_13_0"
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
                    "Reshape_12",
                    "Reshape_13"
                ],
                "name": "Dot_14",
                "op": "Dot",
                "outputs": [
                    "Dot_14_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Dot_14"
                ],
                "name": "Reshape_15",
                "op": "Reshape",
                "output_shape": [
                    100,
                    2,
                    4
                ],
                "outputs": [
                    "Reshape_15_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_15"
                ],
                "lower_bounds": [
                    0,
                    0,
                    0
                ],
                "name": "Slice_16",
                "op": "Slice",
                "outputs": [
                    "Slice_16_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    100,
                    2,
                    1
                ]
            },
            {
                "inputs": [
                    "Reshape_15"
                ],
                "lower_bounds": [
                    0,
                    0,
                    1
                ],
                "name": "Slice_27",
                "op": "Slice",
                "outputs": [
                    "Slice_27_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    100,
                    2,
                    2
                ]
            },
            {
                "inputs": [
                    "Reshape_15"
                ],
                "lower_bounds": [
                    0,
                    0,
                    2
                ],
                "name": "Slice_38",
                "op": "Slice",
                "outputs": [
                    "Slice_38_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    100,
                    2,
                    3
                ]
            },
            {
                "inputs": [
                    "Reshape_15"
                ],
                "lower_bounds": [
                    0,
                    0,
                    3
                ],
                "name": "Slice_49",
                "op": "Slice",
                "outputs": [
                    "Slice_49_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    100,
                    2,
                    4
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_16"
                ],
                "name": "Reshape_17",
                "op": "Reshape",
                "output_shape": [
                    100,
                    2
                ],
                "outputs": [
                    "Reshape_17_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_27"
                ],
                "name": "Reshape_28",
                "op": "Reshape",
                "output_shape": [
                    100,
                    2
                ],
                "outputs": [
                    "Reshape_28_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_38"
                ],
                "name": "Reshape_39",
                "op": "Reshape",
                "output_shape": [
                    100,
                    2
                ],
                "outputs": [
                    "Reshape_39_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_49"
                ],
                "name": "Reshape_50",
                "op": "Reshape",
                "output_shape": [
                    100,
                    2
                ],
                "outputs": [
                    "Reshape_50_0"
                ]
            },
            {
                "inputs": [
                    "Dot_4",
                    "Reshape_17"
                ],
                "name": "Add_18",
                "op": "Add",
                "outputs": [
                    "Add_18_0"
                ]
            },
            {
                "inputs": [
                    "Add_18",
                    "Broadcast_6"
                ],
                "name": "Add_19",
                "op": "Add",
                "outputs": [
                    "Add_19_0"
                ]
            },
            {
                "inputs": [
                    "Add_19"
                ],
                "name": "Sum_20",
                "op": "Sum",
                "outputs": [
                    "Sum_20_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "inputs": [
                    "Sum_20",
                    "Broadcast_8"
                ],
                "name": "Multiply_21",
                "op": "Multiply",
                "outputs": [
                    "Multiply_21_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Multiply_21"
                ],
                "name": "Broadcast_22",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_22_0"
                ],
                "shape": [
                    100,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_19",
                    "Broadcast_22"
                ],
                "name": "Subtract_23",
                "op": "Subtract",
                "outputs": [
                    "Subtract_23_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_23",
                    "Subtract_23"
                ],
                "name": "Multiply_24",
                "op": "Multiply",
                "outputs": [
                    "Multiply_24_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Multiply_24"
                ],
                "name": "Broadcast_25",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_25_0"
                ],
                "shape": [
                    100,
                    1,
                    2
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Multiply_24"
                ],
                "name": "Dot_26",
                "op": "Dot",
                "outputs": [
                    "Dot_26_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_26",
                    "Reshape_28"
                ],
                "name": "Add_29",
                "op": "Add",
                "outputs": [
                    "Add_29_0"
                ]
            },
            {
                "inputs": [
                    "Add_29",
                    "Broadcast_6"
                ],
                "name": "Add_30",
                "op": "Add",
                "outputs": [
                    "Add_30_0"
                ]
            },
            {
                "inputs": [
                    "Add_30"
                ],
                "name": "Sum_31",
                "op": "Sum",
                "outputs": [
                    "Sum_31_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "inputs": [
                    "Sum_31",
                    "Broadcast_8"
                ],
                "name": "Multiply_32",
                "op": "Multiply",
                "outputs": [
                    "Multiply_32_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Multiply_32"
                ],
                "name": "Broadcast_33",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_33_0"
                ],
                "shape": [
                    100,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_30",
                    "Broadcast_33"
                ],
                "name": "Subtract_34",
                "op": "Subtract",
                "outputs": [
                    "Subtract_34_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_34",
                    "Subtract_34"
                ],
                "name": "Multiply_35",
                "op": "Multiply",
                "outputs": [
                    "Multiply_35_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Multiply_35"
                ],
                "name": "Broadcast_36",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_36_0"
                ],
                "shape": [
                    100,
                    1,
                    2
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Multiply_35"
                ],
                "name": "Dot_37",
                "op": "Dot",
                "outputs": [
                    "Dot_37_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_37",
                    "Reshape_39"
                ],
                "name": "Add_40",
                "op": "Add",
                "outputs": [
                    "Add_40_0"
                ]
            },
            {
                "inputs": [
                    "Add_40",
                    "Broadcast_6"
                ],
                "name": "Add_41",
                "op": "Add",
                "outputs": [
                    "Add_41_0"
                ]
            },
            {
                "inputs": [
                    "Add_41"
                ],
                "name": "Sum_42",
                "op": "Sum",
                "outputs": [
                    "Sum_42_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "inputs": [
                    "Sum_42",
                    "Broadcast_8"
                ],
                "name": "Multiply_43",
                "op": "Multiply",
                "outputs": [
                    "Multiply_43_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Multiply_43"
                ],
                "name": "Broadcast_44",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_44_0"
                ],
                "shape": [
                    100,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_41",
                    "Broadcast_44"
                ],
                "name": "Subtract_45",
                "op": "Subtract",
                "outputs": [
                    "Subtract_45_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_45",
                    "Subtract_45"
                ],
                "name": "Multiply_46",
                "op": "Multiply",
                "outputs": [
                    "Multiply_46_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Multiply_46"
                ],
                "name": "Broadcast_47",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_47_0"
                ],
                "shape": [
                    100,
                    1,
                    2
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Multiply_46"
                ],
                "name": "Dot_48",
                "op": "Dot",
                "outputs": [
                    "Dot_48_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_48",
                    "Reshape_50"
                ],
                "name": "Add_51",
                "op": "Add",
                "outputs": [
                    "Add_51_0"
                ]
            },
            {
                "inputs": [
                    "Add_51",
                    "Broadcast_6"
                ],
                "name": "Add_52",
                "op": "Add",
                "outputs": [
                    "Add_52_0"
                ]
            },
            {
                "inputs": [
                    "Add_52"
                ],
                "name": "Sum_53",
                "op": "Sum",
                "outputs": [
                    "Sum_53_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "inputs": [
                    "Sum_53",
                    "Broadcast_8"
                ],
                "name": "Multiply_54",
                "op": "Multiply",
                "outputs": [
                    "Multiply_54_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Multiply_54"
                ],
                "name": "Broadcast_55",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_55_0"
                ],
                "shape": [
                    100,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_52",
                    "Broadcast_55"
                ],
                "name": "Subtract_56",
                "op": "Subtract",
                "outputs": [
                    "Subtract_56_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_56",
                    "Subtract_56"
                ],
                "name": "Multiply_57",
                "op": "Multiply",
                "outputs": [
                    "Multiply_57_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Multiply_57"
                ],
                "name": "Broadcast_58",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_58_0"
                ],
                "shape": [
                    100,
                    1,
                    2
                ]
            },
            {
                "axis": 1,
                "inputs": [
                    "Broadcast_25",
                    "Broadcast_36",
                    "Broadcast_47",
                    "Broadcast_58"
                ],
                "name": "Concat_59",
                "op": "Concat",
                "outputs": [
                    "Concat_59_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Concat_59"
                ],
                "name": "Reshape_61",
                "op": "Reshape",
                "output_shape": [
                    100,
                    8
                ],
                "outputs": [
                    "Reshape_61_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_60",
                    "Reshape_61"
                ],
                "name": "Dot_62",
                "op": "Dot",
                "outputs": [
                    "Dot_62_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Dot_62"
                ],
                "name": "Reshape_63",
                "op": "Reshape",
                "output_shape": [
                    50,
                    4,
                    2
                ],
                "outputs": [
                    "Reshape_63_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_63",
                    "Broadcast_65"
                ],
                "name": "Add_66",
                "op": "Add",
                "outputs": [
                    "Add_66_0"
                ]
            },
            {
                "inputs": [
                    "Add_66"
                ],
                "name": "Result_68",
                "op": "Result",
                "outputs": [
                    "Result_68_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_10",
            "Parameter_67",
            "Parameter_0",
            "Parameter_1",
            "Parameter_5",
            "Parameter_9",
            "Parameter_64"
        ],
        "result": [
            "Result_68"
        ]
    }
]
