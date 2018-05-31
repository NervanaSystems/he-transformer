[
    {
        "name": "Function_10",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1267",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1267_0"
                ],
                "shape": [
                    5,
                    784
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1266",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1266_0"
                ],
                "shape": [
                    10
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1265",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1265_0"
                ],
                "shape": [
                    784,
                    10
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Parameter_1266"
                ],
                "name": "Broadcast_1269",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_1269_0"
                ],
                "shape": [
                    5,
                    10
                ]
            },
            {
                "inputs": [
                    "Parameter_1267",
                    "Parameter_1265"
                ],
                "name": "Dot_1268",
                "op": "Dot",
                "outputs": [
                    "Dot_1268_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_1268",
                    "Broadcast_1269"
                ],
                "name": "Add_1270",
                "op": "Add",
                "outputs": [
                    "Add_1270_0"
                ]
            },
            {
                "inputs": [
                    "Add_1270"
                ],
                "name": "Result_1273",
                "op": "Result",
                "outputs": [
                    "Result_1273_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_1265",
            "Parameter_1266",
            "Parameter_1267"
        ],
        "result": [
            "Result_1273"
        ]
    }
]