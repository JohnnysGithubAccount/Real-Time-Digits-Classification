// clearing the canvas
const canvas = document.getElementById('canvas')
const ctx = canvas.getContext('2d')
const predictionDisplay = document.querySelector(".result-prediction h1");

let isDrawing = false

window.addEventListener("load", () => {
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight
})

const startDrawing = () => {
    isDrawing = true
}

const drawing = (e) => {
    if (!isDrawing) return 
    if (e.buttons !== 1) return
    ctx.beginPath()
    ctx.lineTo(e.offsetX, e.offsetY)  
    ctx.strokeStyle = '#FFFFFF'
    ctx.lineWidth = 50
    ctx.lineCap = 'round'
    ctx.stroke()  
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width,canvas.height)
    predictionDisplay.textContent = "Predict: "
}

function whenMouseUp() {
    isDrawing = false;

    const dataURL = canvas.toDataURL("image/png").split(',')[1]; // This gives you a base64-encoded image
    const dataToSend = {
        "image": dataURL
    };
    fetch("/predict", {
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(dataToSend)
    })
        .then(response => response.json())
        .then(data => predictionDisplay.textContent = data.message)
        .catch(error => console.error("Error:", error))
}

canvas.addEventListener("mousedown", startDrawing)
canvas.addEventListener('mousemove', drawing)
canvas.addEventListener("mouseup", whenMouseUp)

// on click flask
function givePrediction() {
    const dataURL = canvas.toDataURL("image/png").split(',')[1]; // This gives you a base64-encoded image
    const dataToSend = {
        "image": dataURL
    };
    fetch("/predict", {
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(dataToSend)
    })
        .then(response => response.json())
        .then(data => predictionDisplay.textContent = data.message)
        .catch(error => console.error("Error:", error)
)}

