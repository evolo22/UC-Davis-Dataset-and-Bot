const chatBody = document.querySelector('.chat-body');
const messageInput = document.querySelector('.message-input');
const chatForm = document.querySelector('.chat-form');
const closeButton = document.getElementById('close-chatbot');

// Generate a unique session ID for this chat session
const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

// Function to scroll chat to bottom
function scrollToBottom() {
    chatBody.scrollTop = chatBody.scrollHeight;
}

// Function to create a bot message element
function createBotMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    messageDiv.innerHTML = `
        <svg class="bot-avatar" xmlns="http://www.w3.org/2000/svg" 
        width="50" height="50" viewBox="0 0 1024 1024">
        </svg>
        <div class="message-text">${text}</div>
    `;
    
    return messageDiv;
}

// Function to create a user message element
function createUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    
    messageDiv.innerHTML = `
        <div class="message-text">${text}</div>
    `;
    
    return messageDiv;
}

// Function to create a thinking indicator
function createThinkingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message thinking';
    
    messageDiv.innerHTML = `
        <svg class="bot-avatar" xmlns="http://www.w3.org/2000/svg" 
        width="50" height="50" viewBox="0 0 1024 1024">
        </svg>
        <div class="message-text">
            <div class="thinking-indicator">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    `;
    
    return messageDiv;
}

// Handle form submission
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const userMessage = messageInput.value.trim();
    if (!userMessage) return;
    
    // Add user message to chat
    const userMessageElement = createUserMessage(userMessage);
    chatBody.appendChild(userMessageElement);
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = '45px';
    
    // Scroll to bottom after adding user message
    scrollToBottom();
    
    // Add thinking indicator
    const thinkingElement = createThinkingIndicator();
    chatBody.appendChild(thinkingElement);
    scrollToBottom();
    
    try {
        // Send message to backend
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                message: userMessage,
                session_id: sessionId 
            })
        });
        
        const data = await response.json();
        
        // Remove thinking indicator
        thinkingElement.remove();
        
        // Add bot response (Flask returns 'reply' not 'response')
        const botMessageElement = createBotMessage(data.reply || "Sorry, I couldn't process that.");
        chatBody.appendChild(botMessageElement);
        
        // Scroll to bottom after adding bot message
        scrollToBottom();
        
    } catch (error) {
        console.error('Error:', error);
        thinkingElement.remove();
        
        const errorMessage = createBotMessage("Sorry, something went wrong. Please try again.");
        chatBody.appendChild(errorMessage);
        scrollToBottom();
    }
});

// Auto-resize textarea as user types
messageInput.addEventListener('input', function() {
    this.style.height = '45px';
    this.style.height = Math.min(this.scrollHeight, 100) + 'px';
});

// Handle Enter key (submit) vs Shift+Enter (new line)
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// Close chatbot button (optional - you can customize this)
if (closeButton) {
    closeButton.addEventListener('click', () => {
        // You can add functionality here, like minimizing the chatbot
        console.log('Close button clicked');
    });
}

// Scroll to bottom on initial load
window.addEventListener('load', scrollToBottom);