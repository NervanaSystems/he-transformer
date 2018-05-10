[
    {
        "name": "Function_0",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_74",
                "op": "Parameter",
                "outputs": [
                    "Parameter_74_0"
                ],
                "shape": [
                    50
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_7",
                "op": "Parameter",
                "outputs": [
                    "Parameter_7_0"
                ],
                "shape": [
                    500,
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
                    500
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
                    500,
                    500
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
                    500
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_77",
                "op": "Parameter",
                "outputs": [
                    "Parameter_77_0"
                ],
                "shape": [
                    50,
                    8
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_8",
                "op": "Parameter",
                "outputs": [
                    "Parameter_8_0"
                ],
                "shape": [
                    50,
                    8
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
                    "Parameter_74"
                ],
                "name": "Broadcast_75",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_75_0"
                ],
                "shape": [
                    50,
                    8,
                    50
                ]
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Parameter_7"
                ],
                "name": "Reshape_10",
                "op": "Reshape",
                "output_shape": [
                    500,
                    50
                ],
                "outputs": [
                    "Reshape_10_0"
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
                    500,
                    50
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
                "name": "Reshape_70",
                "op": "Reshape",
                "output_shape": [
                    50,
                    500
                ],
                "outputs": [
                    "Reshape_70_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_8"
                ],
                "name": "OneHot_9",
                "one_hot_axis": 0,
                "op": "OneHot",
                "outputs": [
                    "OneHot_9_0"
                ],
                "shape": [
                    50,
                    50,
                    8
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
                    500,
                    50
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "OneHot_9"
                ],
                "name": "Reshape_11",
                "op": "Reshape",
                "output_shape": [
                    50,
                    400
                ],
                "outputs": [
                    "Reshape_11_0"
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
                    "Reshape_10",
                    "Reshape_11"
                ],
                "name": "Dot_12",
                "op": "Dot",
                "outputs": [
                    "Dot_12_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Dot_12"
                ],
                "name": "Reshape_13",
                "op": "Reshape",
                "output_shape": [
                    500,
                    50,
                    8
                ],
                "outputs": [
                    "Reshape_13_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_13"
                ],
                "lower_bounds": [
                    0,
                    0,
                    0
                ],
                "name": "Slice_14",
                "op": "Slice",
                "outputs": [
                    "Slice_14_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    50,
                    1
                ]
            },
            {
                "inputs": [
                    "Reshape_13"
                ],
                "lower_bounds": [
                    0,
                    0,
                    1
                ],
                "name": "Slice_21",
                "op": "Slice",
                "outputs": [
                    "Slice_21_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    50,
                    2
                ]
            },
            {
                "inputs": [
                    "Reshape_13"
                ],
                "lower_bounds": [
                    0,
                    0,
                    2
                ],
                "name": "Slice_28",
                "op": "Slice",
                "outputs": [
                    "Slice_28_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    50,
                    3
                ]
            },
            {
                "inputs": [
                    "Reshape_13"
                ],
                "lower_bounds": [
                    0,
                    0,
                    3
                ],
                "name": "Slice_35",
                "op": "Slice",
                "outputs": [
                    "Slice_35_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    50,
                    4
                ]
            },
            {
                "inputs": [
                    "Reshape_13"
                ],
                "lower_bounds": [
                    0,
                    0,
                    4
                ],
                "name": "Slice_42",
                "op": "Slice",
                "outputs": [
                    "Slice_42_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    50,
                    5
                ]
            },
            {
                "inputs": [
                    "Reshape_13"
                ],
                "lower_bounds": [
                    0,
                    0,
                    5
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
                    500,
                    50,
                    6
                ]
            },
            {
                "inputs": [
                    "Reshape_13"
                ],
                "lower_bounds": [
                    0,
                    0,
                    6
                ],
                "name": "Slice_56",
                "op": "Slice",
                "outputs": [
                    "Slice_56_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    50,
                    7
                ]
            },
            {
                "inputs": [
                    "Reshape_13"
                ],
                "lower_bounds": [
                    0,
                    0,
                    7
                ],
                "name": "Slice_63",
                "op": "Slice",
                "outputs": [
                    "Slice_63_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    50,
                    8
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_14"
                ],
                "name": "Reshape_15",
                "op": "Reshape",
                "output_shape": [
                    500,
                    50
                ],
                "outputs": [
                    "Reshape_15_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_21"
                ],
                "name": "Reshape_22",
                "op": "Reshape",
                "output_shape": [
                    500,
                    50
                ],
                "outputs": [
                    "Reshape_22_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_28"
                ],
                "name": "Reshape_29",
                "op": "Reshape",
                "output_shape": [
                    500,
                    50
                ],
                "outputs": [
                    "Reshape_29_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_35"
                ],
                "name": "Reshape_36",
                "op": "Reshape",
                "output_shape": [
                    500,
                    50
                ],
                "outputs": [
                    "Reshape_36_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_42"
                ],
                "name": "Reshape_43",
                "op": "Reshape",
                "output_shape": [
                    500,
                    50
                ],
                "outputs": [
                    "Reshape_43_0"
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
                    500,
                    50
                ],
                "outputs": [
                    "Reshape_50_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_56"
                ],
                "name": "Reshape_57",
                "op": "Reshape",
                "output_shape": [
                    500,
                    50
                ],
                "outputs": [
                    "Reshape_57_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_63"
                ],
                "name": "Reshape_64",
                "op": "Reshape",
                "output_shape": [
                    500,
                    50
                ],
                "outputs": [
                    "Reshape_64_0"
                ]
            },
            {
                "inputs": [
                    "Dot_4",
                    "Reshape_15"
                ],
                "name": "Add_16",
                "op": "Add",
                "outputs": [
                    "Add_16_0"
                ]
            },
            {
                "inputs": [
                    "Add_16",
                    "Broadcast_6"
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
                    500,
                    1,
                    50
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Tanh_18"
                ],
                "name": "Dot_20",
                "op": "Dot",
                "outputs": [
                    "Dot_20_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_20",
                    "Reshape_22"
                ],
                "name": "Add_23",
                "op": "Add",
                "outputs": [
                    "Add_23_0"
                ]
            },
            {
                "inputs": [
                    "Add_23",
                    "Broadcast_6"
                ],
                "name": "Add_24",
                "op": "Add",
                "outputs": [
                    "Add_24_0"
                ]
            },
            {
                "inputs": [
                    "Add_24"
                ],
                "name": "Tanh_25",
                "op": "Tanh",
                "outputs": [
                    "Tanh_25_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Tanh_25"
                ],
                "name": "Broadcast_26",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_26_0"
                ],
                "shape": [
                    500,
                    1,
                    50
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Tanh_25"
                ],
                "name": "Dot_27",
                "op": "Dot",
                "outputs": [
                    "Dot_27_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_27",
                    "Reshape_29"
                ],
                "name": "Add_30",
                "op": "Add",
                "outputs": [
                    "Add_30_0"
                ]
            },
            {
                "inputs": [
                    "Add_30",
                    "Broadcast_6"
                ],
                "name": "Add_31",
                "op": "Add",
                "outputs": [
                    "Add_31_0"
                ]
            },
            {
                "inputs": [
                    "Add_31"
                ],
                "name": "Tanh_32",
                "op": "Tanh",
                "outputs": [
                    "Tanh_32_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Tanh_32"
                ],
                "name": "Broadcast_33",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_33_0"
                ],
                "shape": [
                    500,
                    1,
                    50
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Tanh_32"
                ],
                "name": "Dot_34",
                "op": "Dot",
                "outputs": [
                    "Dot_34_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_34",
                    "Reshape_36"
                ],
                "name": "Add_37",
                "op": "Add",
                "outputs": [
                    "Add_37_0"
                ]
            },
            {
                "inputs": [
                    "Add_37",
                    "Broadcast_6"
                ],
                "name": "Add_38",
                "op": "Add",
                "outputs": [
                    "Add_38_0"
                ]
            },
            {
                "inputs": [
                    "Add_38"
                ],
                "name": "Tanh_39",
                "op": "Tanh",
                "outputs": [
                    "Tanh_39_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Tanh_39"
                ],
                "name": "Broadcast_40",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_40_0"
                ],
                "shape": [
                    500,
                    1,
                    50
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Tanh_39"
                ],
                "name": "Dot_41",
                "op": "Dot",
                "outputs": [
                    "Dot_41_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_41",
                    "Reshape_43"
                ],
                "name": "Add_44",
                "op": "Add",
                "outputs": [
                    "Add_44_0"
                ]
            },
            {
                "inputs": [
                    "Add_44",
                    "Broadcast_6"
                ],
                "name": "Add_45",
                "op": "Add",
                "outputs": [
                    "Add_45_0"
                ]
            },
            {
                "inputs": [
                    "Add_45"
                ],
                "name": "Tanh_46",
                "op": "Tanh",
                "outputs": [
                    "Tanh_46_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Tanh_46"
                ],
                "name": "Broadcast_47",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_47_0"
                ],
                "shape": [
                    500,
                    1,
                    50
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Tanh_46"
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
                "name": "Tanh_53",
                "op": "Tanh",
                "outputs": [
                    "Tanh_53_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Tanh_53"
                ],
                "name": "Broadcast_54",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_54_0"
                ],
                "shape": [
                    500,
                    1,
                    50
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Tanh_53"
                ],
                "name": "Dot_55",
                "op": "Dot",
                "outputs": [
                    "Dot_55_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_55",
                    "Reshape_57"
                ],
                "name": "Add_58",
                "op": "Add",
                "outputs": [
                    "Add_58_0"
                ]
            },
            {
                "inputs": [
                    "Add_58",
                    "Broadcast_6"
                ],
                "name": "Add_59",
                "op": "Add",
                "outputs": [
                    "Add_59_0"
                ]
            },
            {
                "inputs": [
                    "Add_59"
                ],
                "name": "Tanh_60",
                "op": "Tanh",
                "outputs": [
                    "Tanh_60_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Tanh_60"
                ],
                "name": "Dot_62",
                "op": "Dot",
                "outputs": [
                    "Dot_62_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Tanh_60"
                ],
                "name": "Broadcast_61",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_61_0"
                ],
                "shape": [
                    500,
                    1,
                    50
                ]
            },
            {
                "inputs": [
                    "Dot_62",
                    "Reshape_64"
                ],
                "name": "Add_65",
                "op": "Add",
                "outputs": [
                    "Add_65_0"
                ]
            },
            {
                "inputs": [
                    "Add_65",
                    "Broadcast_6"
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
                "name": "Tanh_67",
                "op": "Tanh",
                "outputs": [
                    "Tanh_67_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Tanh_67"
                ],
                "name": "Broadcast_68",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_68_0"
                ],
                "shape": [
                    500,
                    1,
                    50
                ]
            },
            {
                "axis": 1,
                "inputs": [
                    "Broadcast_19",
                    "Broadcast_26",
                    "Broadcast_33",
                    "Broadcast_40",
                    "Broadcast_47",
                    "Broadcast_54",
                    "Broadcast_61",
                    "Broadcast_68"
                ],
                "name": "Concat_69",
                "op": "Concat",
                "outputs": [
                    "Concat_69_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Concat_69"
                ],
                "name": "Reshape_71",
                "op": "Reshape",
                "output_shape": [
                    500,
                    400
                ],
                "outputs": [
                    "Reshape_71_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_70",
                    "Reshape_71"
                ],
                "name": "Dot_72",
                "op": "Dot",
                "outputs": [
                    "Dot_72_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Dot_72"
                ],
                "name": "Reshape_73",
                "op": "Reshape",
                "output_shape": [
                    50,
                    8,
                    50
                ],
                "outputs": [
                    "Reshape_73_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_73",
                    "Broadcast_75"
                ],
                "name": "Add_76",
                "op": "Add",
                "outputs": [
                    "Add_76_0"
                ]
            },
            {
                "inputs": [
                    "Add_76"
                ],
                "name": "Result_78",
                "op": "Result",
                "outputs": [
                    "Result_78_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_8",
            "Parameter_77",
            "Parameter_0",
            "Parameter_1",
            "Parameter_5",
            "Parameter_7",
            "Parameter_74"
        ],
        "result": [
            "Result_78"
        ]
    }
]
