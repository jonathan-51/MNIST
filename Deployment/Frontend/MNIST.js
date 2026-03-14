/* =========================================================
   CANVAS INITIALIZATION
   ========================================================= */

// Get reference to the canvas element and 2D drawing context
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

canvas.width = canvas.offsetWidth;
canvas.height = canvas.offsetHeight;

// Configure Drawing Style
ctx.strokeStyle = "white";
ctx.lineWidth = 30;
ctx.lineCap = "round";

// Setting up Drawing State
let isDrawing = false;
let lastX = 0;
let lastY = 0;

/* =========================================================
   DRAWING FUNCTIONS
   ========================================================= */

//=============================== Desktop ===============================//

// Start drawing when mouse is pressed down
canvas.addEventListener("mousedown",(e)=> {
    isDrawing = true;
    [lastX,lastY] = [e.offsetX,e.offsetY]; // Record initial mouse position
});

// Draw as the mouse moves
canvas.addEventListener("mousemove",(e) => {
    if (!isDrawing) return; // Skips if mouse is not pressed down

    // Occurs even for the slightest movement
    ctx.beginPath(); // Initializes Drawing a line
    ctx.moveTo(lastX,lastY); // Moves virtual pen to starting position
    ctx.lineTo(e.offsetX,e.offsetY); // Moves virtual pen to the pens new position after moving
    ctx.stroke(); // Draws the line between the two positions

    // Updates most recent position, so line is drawn from original position
    [lastX,lastY] = [e.offsetX,e.offsetY];
});

// Stops drawing when the mouse is released or moves out of the canvas (Reintializes the variables essentially)
canvas.addEventListener("mouseup", () => isDrawing = false);
canvas.addEventListener("mouseout", () => isDrawing = false);

//=============================== Mobile ===============================//

// Start drawing when mouse is pressed down
canvas.addEventListener("touchstart",(e)=> {
    e.preventDefault(); // stop page scroll
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();

    lastX = e.touches[0].clientX - rect.left;
    lastY = e.touches[0].clientY - rect.top;


});

// Draw as the mouse moves
canvas.addEventListener("touchmove",(e) => {
    e.preventDefault(); // stop page scroll
    if (!isDrawing) return; // Skips if mouse is not pressed down

    const rect = canvas.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const y = e.touches[0].clientY - rect.top;

    // Occurs even for the slightest movement
    ctx.beginPath(); // Initializes Drawing a line
    ctx.moveTo(lastX,lastY); // Moves virtual pen to initial position
    ctx.lineTo(x, y); // Moves virtual pen to the pens new position after moving
    ctx.stroke(); // Draws the line between the two positions

    // Updates most recent position, so line is drawn from original position
    [lastX,lastY] = [x,y];
});

// Stops drawing when the mouse is released or moves out of the canvas (Reintializes the variables essentially)
canvas.addEventListener("touchend", () => isDrawing = false);
canvas.addEventListener("touchcancel", () => isDrawing = false);

/* =========================================================
   RESET BUTTON
   ========================================================= */

// Getting Reset Button Element
const reset = document.getElementById("reset");

// Clears Canvas when button is clicked
reset.addEventListener("click",()=> {
    // First arguments establish starting position, last 2 arguments establish final position. The box formed is the area being cleared
    ctx.clearRect(0,0,canvas.width,canvas.height)
})

/* =========================================================
   PREDICT BUTTON
   ========================================================= */

// Getting Predict Button Element
const predict = document.getElementById("predict");
// Getting Fill Animation Element
const fill = predict.querySelector(".fill");

// Function that handles the canvas image data when a blob is passed through
function callback(blob) {
    // Creating an empty FormData object that will house all the metadata pertaining to the data in the form of a key-value pair
    const formData = new FormData();
    // Converts blob into a png file
    const file = new File([blob],"digit.png",{type:"image/png"})
    // Storing file in to FormData object
    formData.append("file",file);

    // Sends image to backend for prediction
    fetch("/predict", {method:"POST", body: formData}) // POST request made at /predict route, sending FormData object as data
        .then(res => res.json()) // When server responds, parse that response as a JSON
        .then(data => {

            const predDiv = document.getElementById("prediction");
            const newPredDiv = predDiv.textContent.slice(0,12);
            predDiv.textContent = `${newPredDiv} ${data.Prediction}`;

            const confDiv = document.getElementById("confidence");
            const newConfDiv = confDiv.textContent.slice(0,12);
            confDiv.textContent = `${newConfDiv} ${data.Confidence}%`;

            for (const key in data.Probabilities) {
                const probDiv = document.getElementById(`num${key}`);
                const newProbDiv = probDiv.textContent.slice(0,2);
                probDiv.textContent = `${newProbDiv} ${data.Probabilities[key]}%`;
            }
        }); // Logs and Displays this data

    return
}

// Processes the digit sketched on the canvas when prediction button is pressed
predict.addEventListener("click",()=> {
    //Disables the button when pressed
    predict.classList.add("predict-disabled");
    //Enables the class responsbile for the fill animation
    fill.classList.add("active");

    // Processes the encoded blob into an image
    canvas.toBlob(callback,"image/png");

    // After 3s, renable button and remove the fill animation.
    setTimeout(()=> {
        predict.classList.remove("predict-disabled");
        fill.classList.remove("active");
    }, 3000)
})
