const chatBody = document.querySelector(".chat-body");
const chatForm = document.querySelector(".chat-form");
const messageInput = document.querySelector(".message-input");

function addMessage(content, className) {
  const message = document.createElement("div");
  message.classList.add("message", className);
  message.innerHTML = `<div class="message-text">${content}</div>`;
  chatBody.appendChild(message);
  chatBody.scrollTop = chatBody.scrollHeight;
}

function addThinkingIndicator() {
  const thinking = document.createElement("div");
  thinking.classList.add("message", "bot-message", "thinking");
  thinking.innerHTML = `
    <svg class="bot-avatar" xmlns="http://www.w3.org/2000/svg" width="35" height="35" viewBox="0 0 1024 1024"></svg>
    <div class="message-text">
      <div class="thinking-indicator">
        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
      </div>
    </div>`;
  chatBody.appendChild(thinking);
  chatBody.scrollTop = chatBody.scrollHeight;
  return thinking;
}

const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const userText = messageInput.value.trim();
  if (!userText) return;

  addMessage(userText, "user-message");
  messageInput.value = "";

  const thinking = addThinkingIndicator();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ 
        message: userText,
        session_id: sessionId
      })  
    });

    const data = await res.json();
    chatBody.removeChild(thinking);

    console.log(`Intent: ${data.intent}, Confidence: ${(data.confidence * 100).toFixed(1)}%`);

    addMessage(data.reply || "Sorry, I didn't catch that.", "bot-message");
  } catch (err) {
    chatBody.removeChild(thinking);
    addMessage("Error: could not reach the server.", "bot-message");
  }
});

// Allow pressing Enter to send
messageInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event("submit"));
  }
});