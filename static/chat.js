// Chat Interface JavaScript
class ChatInterface {
    constructor() {
        this.conversationId = null;
        this.messageCount = 0;
        this.totalMemories = 0;

        // DOM elements
        this.messagesContainer = document.getElementById('messages');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.clearBtn = document.getElementById('clear-btn');
        this.typingIndicator = document.getElementById('typing-indicator');
        this.statusText = document.getElementById('status-text');
        this.statusDot = document.querySelector('.status-dot');

        // Initialize
        this.init();
    }

    init() {
        // Event listeners
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        if (this.clearBtn) {
            this.clearBtn.addEventListener('click', () => this.clearConversation());
        }

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = this.messageInput.scrollHeight + 'px';
        });

        // Check server health
        this.checkHealth();

        // Update stats periodically
        this.updateStats();
        setInterval(() => this.updateStats(), 5000);
    }

    async checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();

            if (data.status === 'healthy') {
                this.setStatus('Connected', true);
            } else {
                this.setStatus('Degraded', false);
            }
        } catch (error) {
            this.setStatus('Disconnected', false);
            console.error('Health check failed:', error);
        }
    }

    setStatus(text, isHealthy) {
        this.statusText.textContent = text;
        if (isHealthy) {
            this.statusDot.classList.remove('disconnected');
        } else {
            this.statusDot.classList.add('disconnected');
        }
    }

    async updateStats() {
        try {
            const response = await fetch('/stats');
            const data = await response.json();

            // Update sidebar stats (desktop)
            const statEntities = document.getElementById('stat-entities');
            const statRelationships = document.getElementById('stat-relationships');
            const statConversations = document.getElementById('stat-conversations');

            if (statEntities) statEntities.textContent = data.database.entities;
            if (statRelationships) statRelationships.textContent = data.database.relationships;
            if (statConversations) statConversations.textContent = data.database.conversations;

            // Update mobile stats
            const mobileStats = document.getElementById('mobile-stats');
            if (mobileStats) {
                mobileStats.textContent = `DB: ${data.database.entities} entities, ${data.database.relationships} relationships`;
            }

            // Update session info
            const memoriesCount = document.getElementById('memories-count');
            const messageCountEl = document.getElementById('message-count');

            if (memoriesCount) memoriesCount.textContent = this.totalMemories;
            if (messageCountEl) messageCountEl.textContent = this.messageCount;

        } catch (error) {
            console.error('Failed to update stats:', error);
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        // Disable input
        this.messageInput.disabled = true;
        this.sendBtn.disabled = true;

        // Display user message
        this.addMessage(message, 'user');

        // Clear input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';

        // Show typing indicator
        this.typingIndicator.style.display = 'flex';
        this.scrollToBottom();

        try {
            // Send to API
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: this.conversationId,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Store conversation ID
            if (data.conversation_id) {
                this.conversationId = data.conversation_id;
            }

            // Clean response (remove channel markers if present)
            let cleanResponse = data.response;
            if (cleanResponse.includes('<|channel|>final<|message|>')) {
                cleanResponse = cleanResponse.split('<|channel|>final<|message|>')[1];
            }
            if (cleanResponse.includes('<|end|>')) {
                cleanResponse = cleanResponse.split('<|end|>')[0];
            }

            // Display assistant response
            this.addMessage(cleanResponse, 'assistant', data.memories_extracted);

            // Update counters
            this.messageCount += 2; // User + assistant
            if (data.memories_extracted) {
                this.totalMemories += data.memories_extracted;
            }

            // Update stats after response
            setTimeout(() => this.updateStats(), 1000);

        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage(
                `Error: ${error.message}. Please check if the server is running.`,
                'system'
            );
        } finally {
            // Hide typing indicator
            this.typingIndicator.style.display = 'none';

            // Re-enable input
            this.messageInput.disabled = false;
            this.sendBtn.disabled = false;
            this.messageInput.focus();
        }
    }

    addMessage(content, type, memoriesExtracted = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        // Convert newlines to <br> for proper display
        const formattedContent = content.replace(/\n/g, '<br>');
        contentDiv.innerHTML = `<p>${formattedContent}</p>`;

        messageDiv.appendChild(contentDiv);

        // Add metadata for assistant messages
        if (type === 'assistant' && memoriesExtracted !== null) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';

            if (memoriesExtracted > 0) {
                const badge = document.createElement('span');
                badge.className = 'memory-badge';
                badge.textContent = `${memoriesExtracted} memories`;
                metaDiv.appendChild(badge);
            }

            const timestamp = document.createElement('span');
            timestamp.textContent = new Date().toLocaleTimeString();
            metaDiv.appendChild(timestamp);

            contentDiv.appendChild(metaDiv);
        }

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    clearConversation() {
        if (!confirm('Clear conversation history? This will reset the chat but memories will remain in the database.')) {
            return;
        }

        // Remove all messages except system message
        const messages = this.messagesContainer.querySelectorAll('.message:not(.system-message)');
        messages.forEach(msg => msg.remove());

        // Reset conversation ID
        this.conversationId = null;
        this.messageCount = 0;

        // Add confirmation message
        this.addMessage('Conversation cleared. Starting fresh!', 'system');

        this.updateStats();
    }

    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 100);
    }
}

// Initialize chat interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new ChatInterface();
});
