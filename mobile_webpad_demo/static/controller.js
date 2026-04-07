const config = window.CONTROLLER_CONFIG || {};
const buttonNames = config.buttonNames || ["up", "down", "left", "right", "a", "b", "start", "select"];

const appTitle = document.getElementById("app-title");
const statusEl = document.getElementById("connection-status");
const helperNote = document.getElementById("helper-note");
const buttons = Array.from(document.querySelectorAll("[data-button]"));

let socket = null;
let reconnectTimer = null;
let assignedLabel = null;
const deviceId = getDeviceId();

const activePointers = new Map();
const heldButtons = new Map(buttonNames.map((button) => [button, false]));

appTitle.textContent = (config.appTitle || "MOBILE WEBPAD DEMO").toUpperCase();
if (helperNote && typeof config.helperNoteHtml === "string" && config.helperNoteHtml) {
  helperNote.innerHTML = config.helperNoteHtml;
}

function socketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.hostname;
  return `${protocol}://${host}:${config.wsPort}?device=${encodeURIComponent(deviceId)}`;
}

function createDeviceId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `webpad-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getDeviceId() {
  const storageKey = "mobile-webpad-device-id";
  try {
    const existing = window.localStorage.getItem(storageKey);
    if (existing) {
      return existing;
    }
    const created = createDeviceId();
    window.localStorage.setItem(storageKey, created);
    return created;
  } catch (error) {
    return createDeviceId();
  }
}

function setStatus(text, className) {
  statusEl.textContent = text;
  statusEl.classList.remove("online", "offline");
  if (className) {
    statusEl.classList.add(className);
  }
}

function onlineStatusText() {
  return assignedLabel ? `${assignedLabel} ONLINE` : "ONLINE";
}

function sendMessage(message) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  socket.send(JSON.stringify(message));
}

function updateButtonVisual(buttonName, active) {
  const element = document.querySelector(`[data-button="${buttonName}"]`);
  if (!element) {
    return;
  }
  element.classList.toggle("is-active", active);
}

function pressButton(buttonName) {
  if (heldButtons.get(buttonName)) {
    return;
  }
  heldButtons.set(buttonName, true);
  updateButtonVisual(buttonName, true);
  sendMessage({ type: "button", button: buttonName, pressed: true });
}

function releaseButton(buttonName) {
  if (!heldButtons.get(buttonName)) {
    return;
  }
  heldButtons.set(buttonName, false);
  updateButtonVisual(buttonName, false);
  sendMessage({ type: "button", button: buttonName, pressed: false });
}

function releasePointer(pointerId) {
  const buttonName = activePointers.get(pointerId);
  if (!buttonName) {
    return;
  }

  activePointers.delete(pointerId);
  const stillHeld = Array.from(activePointers.values()).includes(buttonName);
  if (!stillHeld) {
    releaseButton(buttonName);
  }
}

function releaseAll(reason = "release_all") {
  activePointers.clear();
  for (const buttonName of buttonNames) {
    if (heldButtons.get(buttonName)) {
      heldButtons.set(buttonName, false);
      updateButtonVisual(buttonName, false);
    }
  }
  sendMessage({ type: "release_all", reason });
}

function connect() {
  setStatus("CONNECTING", "offline");
  socket = new WebSocket(socketUrl());

  socket.addEventListener("open", () => {
    assignedLabel = null;
    setStatus("ONLINE", "online");
    sendMessage({
      type: "hello",
      deviceId,
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
    });
  });

  socket.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "hello") {
        assignedLabel = message.label || null;
        if (Array.isArray(message.buttons)) {
          for (const buttonName of message.buttons) {
            if (!heldButtons.has(buttonName)) {
              heldButtons.set(buttonName, false);
            }
          }
        }
        setStatus(onlineStatusText(), "online");
      }
    } catch (error) {
      return;
    }
  });

  socket.addEventListener("close", () => {
    assignedLabel = null;
    setStatus("RECONNECTING", "offline");
    releaseAll("socket_closed");
    window.clearTimeout(reconnectTimer);
    reconnectTimer = window.setTimeout(connect, 800);
  });

  socket.addEventListener("error", () => {
    setStatus("ERROR", "offline");
  });
}

for (const element of buttons) {
  element.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    const buttonName = element.dataset.button;
    activePointers.set(event.pointerId, buttonName);
    element.setPointerCapture(event.pointerId);
    pressButton(buttonName);
  });

  const onRelease = (event) => {
    event.preventDefault();
    releasePointer(event.pointerId);
  };

  element.addEventListener("pointerup", onRelease);
  element.addEventListener("pointercancel", onRelease);
  element.addEventListener("lostpointercapture", onRelease);
}

window.addEventListener("blur", () => releaseAll("window_blur"));
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") {
    releaseAll("visibility_change");
  }
});

window.addEventListener("contextmenu", (event) => event.preventDefault());
connect();
