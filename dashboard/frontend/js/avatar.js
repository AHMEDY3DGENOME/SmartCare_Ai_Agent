class CareSenseAvatar {

    constructor() {
        this.avatar = document.getElementById("aiAvatar");
        this.status = document.getElementById("avatarStatus");
    }

    setListening() {
        this.avatar.className = "avatar listening";
        this.status.innerText = "🎤 Listening...";
    }

    setThinking() {
        this.avatar.className = "avatar thinking";
        this.status.innerText = "🧠 Thinking...";
    }

    setSpeaking() {
        this.avatar.className = "avatar speaking";
        this.status.innerText = "🔊 Speaking...";
    }

    setIdle() {
        this.avatar.className = "avatar idle";
        this.status.innerText = "✅ Ready";
    }
}

const careSenseAvatar = new CareSenseAvatar();