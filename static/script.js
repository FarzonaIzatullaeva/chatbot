const chatForm = document.getElementById("chat-form");
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");



window.addEventListener("load", () => {
    appendMessage("Hi! I'm Sam. How can I help you today?", "bot-message");
});


chatForm.addEventListener("submit", (e) => {
    e.preventDefault(); // Prevents page from reloading

    const message = userInput.value;
    if (!message) return; // Don't send empty messages

    // Display user message
    appendMessage(message, "user-message");
    userInput.value = ""; // Clear the input field

    // Scroll to the bottom of the chat box
    chatBox.scrollTop = chatBox.scrollHeight;

    // Send message to the backend
    fetch("/get_response", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: message }),
    })
        .then((response) => response.json())
        .then((data) => {
            // Display bot message
            appendMessage(data.answer, "bot-message");

            // Scroll to the bottom after bot response
            chatBox.scrollTop = chatBox.scrollHeight;
        })
        .catch((error) => {
            console.error("Error:", error);
            appendMessage("Sorry, something went wrong.", "bot-message");
        });
});

// Function to create and append a new message element
function appendMessage(message, className) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", className);
    messageDiv.textContent = message;
    chatBox.appendChild(messageDiv);
}