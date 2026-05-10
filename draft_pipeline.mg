{
    "header": {
        "pipelineVersion": "2.2",
        "releaseVersion": ""
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "inputs": {}
        },
        "FeatureExtraction_1": {
            "nodeType": "FeatureExtraction",
            "inputs": { "input": "{CameraInit_1.output}" }
        },
        "ImageMatching_1": {
            "nodeType": "ImageMatching",
            "inputs": { "input": "{CameraInit_1.output}", "featuresFolders": [ "{FeatureExtraction_1.output}" ] }
        },
        "FeatureMatching_1": {
            "nodeType": "FeatureMatching",
            "inputs": { "input": "{CameraInit_1.output}", "featuresFolders": [ "{FeatureExtraction_1.output}" ], "imagePairsList": "{ImageMatching_1.output}" }
        },
        "StructureFromMotion_1": {
            "nodeType": "StructureFromMotion",
            "inputs": { "input": "{CameraInit_1.output}", "featuresFolders": [ "{FeatureExtraction_1.output}" ], "matchesFolders": [ "{FeatureMatching_1.output}" ] }
        },
        "PrepareDenseScene_1": {
            "nodeType": "PrepareDenseScene",
            "inputs": { "input": "{StructureFromMotion_1.output}" }
        },
        "Meshing_1": {
            "nodeType": "Meshing",
            "inputs": { "input": "{StructureFromMotion_1.output}", "imagesFolder": "{PrepareDenseScene_1.output}" }
        },
        "MeshFiltering_1": {
            "nodeType": "MeshFiltering",
            "inputs": { "inputMesh": "{Meshing_1.output}" }
        },
        "Texturing_1": {
            "nodeType": "Texturing",
            "inputs": { "inputMesh": "{MeshFiltering_1.output}", "inputImages": "{PrepareDenseScene_1.output}" }
        },
        "Publish_1": {
            "nodeType": "Publish",
            "inputs": { 
                "inputFiles": [ "{Texturing_1.output}" ], 
                "output": "/output" 
            }
        }
    }
}