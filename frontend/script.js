function send() {
    const input = document.getElementById("userInput");
    const msg = input.value.trim();
    if (!msg) return;

    const chatBody = document.getElementById("chatBody");

    // User message
    chatBody.innerHTML += `<div class="user-msg">${msg}</div>`;
    input.value = "";

    // Typing indicator
    const typing = document.createElement("div");
    typing.className = "bot-msg";
    typing.innerText = "Typing...";
    chatBody.appendChild(typing);

    // Call backend
    fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
    })
    .then(res => res.json())
    .then(data => {
        typing.remove();

        // ===== If bot returns a syllabus file =====
        if (data.file) {
            chatBody.innerHTML += `<div class="bot-msg">
                ${data.reply} <br>
                <a href="http://127.0.0.1:5000${data.file}" target="_blank" download>⬇ Download Syllabus</a>
            </div>`;
        } else {
            chatBody.innerHTML += `<div class="bot-msg">${data.reply}</div>`;
        }

        // Scroll to bottom
        chatBody.scrollTop = chatBody.scrollHeight;
    })
    .catch(() => {
        typing.innerText = "Server Error";
    });
}

// Send message on Enter key
document.addEventListener("keydown", e => {
    if (e.key === "Enter") send();
});
