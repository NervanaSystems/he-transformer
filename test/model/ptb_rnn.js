[
    {
        "name": "Function_0",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_110",
                "op": "Parameter",
                "outputs": [
                    "Parameter_110_0"
                ],
                "shape": [
                    50
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_11",
                "op": "Parameter",
                "outputs": [
                    "Parameter_11_0"
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
                "name": "Parameter_113",
                "op": "Parameter",
                "outputs": [
                    "Parameter_113_0"
                ],
                "shape": [
                    2,
                    8
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_12",
                "op": "Parameter",
                "outputs": [
                    "Parameter_12_0"
                ],
                "shape": [
                    2,
                    8
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_9",
                "op": "Constant",
                "outputs": [
                    "Constant_9_0"
                ],
                "shape": [],
                "value": [
                    "2"
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
                    "0.002"
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
                    "Parameter_110"
                ],
                "name": "Broadcast_111",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_111_0"
                ],
                "shape": [
                    50,
                    8,
                    2
                ]
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Parameter_11"
                ],
                "name": "Reshape_14",
                "op": "Reshape",
                "output_shape": [
                    500,
                    50
                ],
                "outputs": [
                    "Reshape_14_0"
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
                "name": "Reshape_106",
                "op": "Reshape",
                "output_shape": [
                    50,
                    500
                ],
                "outputs": [
                    "Reshape_106_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_12"
                ],
                "name": "OneHot_13",
                "one_hot_axis": 0,
                "op": "OneHot",
                "outputs": [
                    "OneHot_13_0"
                ],
                "shape": [
                    50,
                    2,
                    8
                ]
            },
            {
                "axes": [
                    0,
                    1
                ],
                "inputs": [
                    "Constant_9"
                ],
                "name": "Broadcast_10",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_10_0"
                ],
                "shape": [
                    500,
                    2
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
                    500,
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
                    "OneHot_13"
                ],
                "name": "Reshape_15",
                "op": "Reshape",
                "output_shape": [
                    50,
                    16
                ],
                "outputs": [
                    "Reshape_15_0"
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
                    "Reshape_14",
                    "Reshape_15"
                ],
                "name": "Dot_16",
                "op": "Dot",
                "outputs": [
                    "Dot_16_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Dot_16"
                ],
                "name": "Reshape_17",
                "op": "Reshape",
                "output_shape": [
                    500,
                    2,
                    8
                ],
                "outputs": [
                    "Reshape_17_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_17"
                ],
                "lower_bounds": [
                    0,
                    0,
                    0
                ],
                "name": "Slice_18",
                "op": "Slice",
                "outputs": [
                    "Slice_18_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    2,
                    1
                ]
            },
            {
                "inputs": [
                    "Reshape_17"
                ],
                "lower_bounds": [
                    0,
                    0,
                    1
                ],
                "name": "Slice_29",
                "op": "Slice",
                "outputs": [
                    "Slice_29_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    2,
                    2
                ]
            },
            {
                "inputs": [
                    "Reshape_17"
                ],
                "lower_bounds": [
                    0,
                    0,
                    2
                ],
                "name": "Slice_40",
                "op": "Slice",
                "outputs": [
                    "Slice_40_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    2,
                    3
                ]
            },
            {
                "inputs": [
                    "Reshape_17"
                ],
                "lower_bounds": [
                    0,
                    0,
                    3
                ],
                "name": "Slice_51",
                "op": "Slice",
                "outputs": [
                    "Slice_51_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    2,
                    4
                ]
            },
            {
                "inputs": [
                    "Reshape_17"
                ],
                "lower_bounds": [
                    0,
                    0,
                    4
                ],
                "name": "Slice_62",
                "op": "Slice",
                "outputs": [
                    "Slice_62_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    2,
                    5
                ]
            },
            {
                "inputs": [
                    "Reshape_17"
                ],
                "lower_bounds": [
                    0,
                    0,
                    5
                ],
                "name": "Slice_73",
                "op": "Slice",
                "outputs": [
                    "Slice_73_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    2,
                    6
                ]
            },
            {
                "inputs": [
                    "Reshape_17"
                ],
                "lower_bounds": [
                    0,
                    0,
                    6
                ],
                "name": "Slice_84",
                "op": "Slice",
                "outputs": [
                    "Slice_84_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    2,
                    7
                ]
            },
            {
                "inputs": [
                    "Reshape_17"
                ],
                "lower_bounds": [
                    0,
                    0,
                    7
                ],
                "name": "Slice_95",
                "op": "Slice",
                "outputs": [
                    "Slice_95_0"
                ],
                "strides": [
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    500,
                    2,
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
                    "Slice_18"
                ],
                "name": "Reshape_19",
                "op": "Reshape",
                "output_shape": [
                    500,
                    2
                ],
                "outputs": [
                    "Reshape_19_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_29"
                ],
                "name": "Reshape_30",
                "op": "Reshape",
                "output_shape": [
                    500,
                    2
                ],
                "outputs": [
                    "Reshape_30_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_40"
                ],
                "name": "Reshape_41",
                "op": "Reshape",
                "output_shape": [
                    500,
                    2
                ],
                "outputs": [
                    "Reshape_41_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_51"
                ],
                "name": "Reshape_52",
                "op": "Reshape",
                "output_shape": [
                    500,
                    2
                ],
                "outputs": [
                    "Reshape_52_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_62"
                ],
                "name": "Reshape_63",
                "op": "Reshape",
                "output_shape": [
                    500,
                    2
                ],
                "outputs": [
                    "Reshape_63_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_73"
                ],
                "name": "Reshape_74",
                "op": "Reshape",
                "output_shape": [
                    500,
                    2
                ],
                "outputs": [
                    "Reshape_74_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_84"
                ],
                "name": "Reshape_85",
                "op": "Reshape",
                "output_shape": [
                    500,
                    2
                ],
                "outputs": [
                    "Reshape_85_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Slice_95"
                ],
                "name": "Reshape_96",
                "op": "Reshape",
                "output_shape": [
                    500,
                    2
                ],
                "outputs": [
                    "Reshape_96_0"
                ]
            },
            {
                "inputs": [
                    "Dot_4",
                    "Reshape_19"
                ],
                "name": "Add_20",
                "op": "Add",
                "outputs": [
                    "Add_20_0"
                ]
            },
            {
                "inputs": [
                    "Add_20",
                    "Broadcast_6"
                ],
                "name": "Add_21",
                "op": "Add",
                "outputs": [
                    "Add_21_0"
                ]
            },
            {
                "inputs": [
                    "Add_21"
                ],
                "name": "Sum_22",
                "op": "Sum",
                "outputs": [
                    "Sum_22_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "inputs": [
                    "Sum_22",
                    "Broadcast_8"
                ],
                "name": "Multiply_23",
                "op": "Multiply",
                "outputs": [
                    "Multiply_23_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Multiply_23"
                ],
                "name": "Broadcast_24",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_24_0"
                ],
                "shape": [
                    500,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_21",
                    "Broadcast_24"
                ],
                "name": "Subtract_25",
                "op": "Subtract",
                "outputs": [
                    "Subtract_25_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_25",
                    "Broadcast_10"
                ],
                "name": "Power_26",
                "op": "Power",
                "outputs": [
                    "Power_26_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Power_26"
                ],
                "name": "Broadcast_27",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_27_0"
                ],
                "shape": [
                    500,
                    1,
                    2
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Power_26"
                ],
                "name": "Dot_28",
                "op": "Dot",
                "outputs": [
                    "Dot_28_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_28",
                    "Reshape_30"
                ],
                "name": "Add_31",
                "op": "Add",
                "outputs": [
                    "Add_31_0"
                ]
            },
            {
                "inputs": [
                    "Add_31",
                    "Broadcast_6"
                ],
                "name": "Add_32",
                "op": "Add",
                "outputs": [
                    "Add_32_0"
                ]
            },
            {
                "inputs": [
                    "Add_32"
                ],
                "name": "Sum_33",
                "op": "Sum",
                "outputs": [
                    "Sum_33_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "inputs": [
                    "Sum_33",
                    "Broadcast_8"
                ],
                "name": "Multiply_34",
                "op": "Multiply",
                "outputs": [
                    "Multiply_34_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Multiply_34"
                ],
                "name": "Broadcast_35",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_35_0"
                ],
                "shape": [
                    500,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_32",
                    "Broadcast_35"
                ],
                "name": "Subtract_36",
                "op": "Subtract",
                "outputs": [
                    "Subtract_36_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_36",
                    "Broadcast_10"
                ],
                "name": "Power_37",
                "op": "Power",
                "outputs": [
                    "Power_37_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Power_37"
                ],
                "name": "Broadcast_38",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_38_0"
                ],
                "shape": [
                    500,
                    1,
                    2
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Power_37"
                ],
                "name": "Dot_39",
                "op": "Dot",
                "outputs": [
                    "Dot_39_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_39",
                    "Reshape_41"
                ],
                "name": "Add_42",
                "op": "Add",
                "outputs": [
                    "Add_42_0"
                ]
            },
            {
                "inputs": [
                    "Add_42",
                    "Broadcast_6"
                ],
                "name": "Add_43",
                "op": "Add",
                "outputs": [
                    "Add_43_0"
                ]
            },
            {
                "inputs": [
                    "Add_43"
                ],
                "name": "Sum_44",
                "op": "Sum",
                "outputs": [
                    "Sum_44_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "inputs": [
                    "Sum_44",
                    "Broadcast_8"
                ],
                "name": "Multiply_45",
                "op": "Multiply",
                "outputs": [
                    "Multiply_45_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Multiply_45"
                ],
                "name": "Broadcast_46",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_46_0"
                ],
                "shape": [
                    500,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_43",
                    "Broadcast_46"
                ],
                "name": "Subtract_47",
                "op": "Subtract",
                "outputs": [
                    "Subtract_47_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_47",
                    "Broadcast_10"
                ],
                "name": "Power_48",
                "op": "Power",
                "outputs": [
                    "Power_48_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Power_48"
                ],
                "name": "Broadcast_49",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_49_0"
                ],
                "shape": [
                    500,
                    1,
                    2
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Power_48"
                ],
                "name": "Dot_50",
                "op": "Dot",
                "outputs": [
                    "Dot_50_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_50",
                    "Reshape_52"
                ],
                "name": "Add_53",
                "op": "Add",
                "outputs": [
                    "Add_53_0"
                ]
            },
            {
                "inputs": [
                    "Add_53",
                    "Broadcast_6"
                ],
                "name": "Add_54",
                "op": "Add",
                "outputs": [
                    "Add_54_0"
                ]
            },
            {
                "inputs": [
                    "Add_54"
                ],
                "name": "Sum_55",
                "op": "Sum",
                "outputs": [
                    "Sum_55_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "inputs": [
                    "Sum_55",
                    "Broadcast_8"
                ],
                "name": "Multiply_56",
                "op": "Multiply",
                "outputs": [
                    "Multiply_56_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Multiply_56"
                ],
                "name": "Broadcast_57",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_57_0"
                ],
                "shape": [
                    500,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_54",
                    "Broadcast_57"
                ],
                "name": "Subtract_58",
                "op": "Subtract",
                "outputs": [
                    "Subtract_58_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_58",
                    "Broadcast_10"
                ],
                "name": "Power_59",
                "op": "Power",
                "outputs": [
                    "Power_59_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Power_59"
                ],
                "name": "Broadcast_60",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_60_0"
                ],
                "shape": [
                    500,
                    1,
                    2
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Power_59"
                ],
                "name": "Dot_61",
                "op": "Dot",
                "outputs": [
                    "Dot_61_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_61",
                    "Reshape_63"
                ],
                "name": "Add_64",
                "op": "Add",
                "outputs": [
                    "Add_64_0"
                ]
            },
            {
                "inputs": [
                    "Add_64",
                    "Broadcast_6"
                ],
                "name": "Add_65",
                "op": "Add",
                "outputs": [
                    "Add_65_0"
                ]
            },
            {
                "inputs": [
                    "Add_65"
                ],
                "name": "Sum_66",
                "op": "Sum",
                "outputs": [
                    "Sum_66_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "inputs": [
                    "Sum_66",
                    "Broadcast_8"
                ],
                "name": "Multiply_67",
                "op": "Multiply",
                "outputs": [
                    "Multiply_67_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Multiply_67"
                ],
                "name": "Broadcast_68",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_68_0"
                ],
                "shape": [
                    500,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_65",
                    "Broadcast_68"
                ],
                "name": "Subtract_69",
                "op": "Subtract",
                "outputs": [
                    "Subtract_69_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_69",
                    "Broadcast_10"
                ],
                "name": "Power_70",
                "op": "Power",
                "outputs": [
                    "Power_70_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Power_70"
                ],
                "name": "Broadcast_71",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_71_0"
                ],
                "shape": [
                    500,
                    1,
                    2
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Power_70"
                ],
                "name": "Dot_72",
                "op": "Dot",
                "outputs": [
                    "Dot_72_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_72",
                    "Reshape_74"
                ],
                "name": "Add_75",
                "op": "Add",
                "outputs": [
                    "Add_75_0"
                ]
            },
            {
                "inputs": [
                    "Add_75",
                    "Broadcast_6"
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
                "name": "Sum_77",
                "op": "Sum",
                "outputs": [
                    "Sum_77_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "inputs": [
                    "Sum_77",
                    "Broadcast_8"
                ],
                "name": "Multiply_78",
                "op": "Multiply",
                "outputs": [
                    "Multiply_78_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Multiply_78"
                ],
                "name": "Broadcast_79",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_79_0"
                ],
                "shape": [
                    500,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_76",
                    "Broadcast_79"
                ],
                "name": "Subtract_80",
                "op": "Subtract",
                "outputs": [
                    "Subtract_80_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_80",
                    "Broadcast_10"
                ],
                "name": "Power_81",
                "op": "Power",
                "outputs": [
                    "Power_81_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Power_81"
                ],
                "name": "Broadcast_82",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_82_0"
                ],
                "shape": [
                    500,
                    1,
                    2
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Power_81"
                ],
                "name": "Dot_83",
                "op": "Dot",
                "outputs": [
                    "Dot_83_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_83",
                    "Reshape_85"
                ],
                "name": "Add_86",
                "op": "Add",
                "outputs": [
                    "Add_86_0"
                ]
            },
            {
                "inputs": [
                    "Add_86",
                    "Broadcast_6"
                ],
                "name": "Add_87",
                "op": "Add",
                "outputs": [
                    "Add_87_0"
                ]
            },
            {
                "inputs": [
                    "Add_87"
                ],
                "name": "Sum_88",
                "op": "Sum",
                "outputs": [
                    "Sum_88_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "inputs": [
                    "Sum_88",
                    "Broadcast_8"
                ],
                "name": "Multiply_89",
                "op": "Multiply",
                "outputs": [
                    "Multiply_89_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Multiply_89"
                ],
                "name": "Broadcast_90",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_90_0"
                ],
                "shape": [
                    500,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_87",
                    "Broadcast_90"
                ],
                "name": "Subtract_91",
                "op": "Subtract",
                "outputs": [
                    "Subtract_91_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_91",
                    "Broadcast_10"
                ],
                "name": "Power_92",
                "op": "Power",
                "outputs": [
                    "Power_92_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Power_92"
                ],
                "name": "Broadcast_93",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_93_0"
                ],
                "shape": [
                    500,
                    1,
                    2
                ]
            },
            {
                "inputs": [
                    "Parameter_1",
                    "Power_92"
                ],
                "name": "Dot_94",
                "op": "Dot",
                "outputs": [
                    "Dot_94_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_94",
                    "Reshape_96"
                ],
                "name": "Add_97",
                "op": "Add",
                "outputs": [
                    "Add_97_0"
                ]
            },
            {
                "inputs": [
                    "Add_97",
                    "Broadcast_6"
                ],
                "name": "Add_98",
                "op": "Add",
                "outputs": [
                    "Add_98_0"
                ]
            },
            {
                "inputs": [
                    "Add_98"
                ],
                "name": "Sum_99",
                "op": "Sum",
                "outputs": [
                    "Sum_99_0"
                ],
                "reduction_axes": [
                    0
                ]
            },
            {
                "inputs": [
                    "Sum_99",
                    "Broadcast_8"
                ],
                "name": "Multiply_100",
                "op": "Multiply",
                "outputs": [
                    "Multiply_100_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Multiply_100"
                ],
                "name": "Broadcast_101",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_101_0"
                ],
                "shape": [
                    500,
                    2
                ]
            },
            {
                "inputs": [
                    "Add_98",
                    "Broadcast_101"
                ],
                "name": "Subtract_102",
                "op": "Subtract",
                "outputs": [
                    "Subtract_102_0"
                ]
            },
            {
                "inputs": [
                    "Subtract_102",
                    "Broadcast_10"
                ],
                "name": "Power_103",
                "op": "Power",
                "outputs": [
                    "Power_103_0"
                ]
            },
            {
                "axes": [
                    1
                ],
                "inputs": [
                    "Power_103"
                ],
                "name": "Broadcast_104",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_104_0"
                ],
                "shape": [
                    500,
                    1,
                    2
                ]
            },
            {
                "axis": 1,
                "inputs": [
                    "Broadcast_27",
                    "Broadcast_38",
                    "Broadcast_49",
                    "Broadcast_60",
                    "Broadcast_71",
                    "Broadcast_82",
                    "Broadcast_93",
                    "Broadcast_104"
                ],
                "name": "Concat_105",
                "op": "Concat",
                "outputs": [
                    "Concat_105_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Concat_105"
                ],
                "name": "Reshape_107",
                "op": "Reshape",
                "output_shape": [
                    500,
                    16
                ],
                "outputs": [
                    "Reshape_107_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_106",
                    "Reshape_107"
                ],
                "name": "Dot_108",
                "op": "Dot",
                "outputs": [
                    "Dot_108_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Dot_108"
                ],
                "name": "Reshape_109",
                "op": "Reshape",
                "output_shape": [
                    50,
                    8,
                    2
                ],
                "outputs": [
                    "Reshape_109_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_109",
                    "Broadcast_111"
                ],
                "name": "Add_112",
                "op": "Add",
                "outputs": [
                    "Add_112_0"
                ]
            },
            {
                "inputs": [
                    "Add_112"
                ],
                "name": "Result_114",
                "op": "Result",
                "outputs": [
                    "Result_114_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_12",
            "Parameter_113",
            "Parameter_0",
            "Parameter_1",
            "Parameter_5",
            "Parameter_11",
            "Parameter_110"
        ],
        "result": [
            "Result_114"
        ]
    }
]
