const problemDropdown = document.getElementById("problem");
const modelDropdown = document.getElementById("model");
const explainableAiDropdown = document.getElementById("explainable_ai");

const modelOptions = {
    // classification: ["VGG16"],
    // object_detection: ["YOLOX", "YOLOv4"],
    segmentation: ["ResNet50"]
};

const explainableAiOptions = {
    // classification: ["GradCAM"],
    // object_detection: ["D-RISE"],
    segmentation: ["GradCAM", "EigenCAM", "HiResCAM"]
};

problemDropdown.addEventListener("change", function () {
    const selectedProblem = problemDropdown.value;

    // Update AI Model dropdown
    modelDropdown.innerHTML = "";
    for (const model of modelOptions[selectedProblem]) {
        const option = document.createElement("option");
        option.text = model;
        option.value = model.toLowerCase();
        modelDropdown.add(option);
    }

    // Update Explainable AI Method dropdown
    explainableAiDropdown.innerHTML = "";
    for (const method of explainableAiOptions[selectedProblem]) {
        const option = document.createElement("option");
        option.text = method;
        option.value = method.toLowerCase();
        explainableAiDropdown.add(option);
    }
});
