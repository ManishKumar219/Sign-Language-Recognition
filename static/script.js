const IMAGE_INTERVAL_MS = 300;
let count = 0;
console.log("Succes");
console.log(IMAGE_INTERVAL_MS)



const startHandDetection = (video, canvas, deviceId) => {
  const socket = new WebSocket('ws://localhost:8000/asl-detection');
  // const socket = new WebSocket('wss://api-for-asl.onrender.com/face-detection');
  let intervalId;

  // Connection opened
  socket.addEventListener('open', function () {

    // Start reading video from device
    navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        deviceId,
        width: { max: 640 },
        height: { max: 480 },
      },
    }).then(function (stream) {
      video.srcObject = stream;
      video.play().then(() => {
        // Adapt overlay canvas size to the video size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Send an image in the WebSocket every 42 ms
        intervalId = setInterval(() => {

          // Create a virtual canvas to draw current video image
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0);

          // Convert it to JPEG and send it to the WebSocket
          canvas.toBlob((blob) => socket.send(blob), 'image/jpeg');
        }, IMAGE_INTERVAL_MS);
      });
    });
  });

  // Listen for messages
  socket.addEventListener('message', function (event) {
    // drawFaceRectangles(video, canvas, JSON.parse(event.data));
    var data = JSON.parse(event.data);
    console.log(count++);
    console.log(data.hands_lms);
    var text_data = document.getElementById('transcribe').innerText;
    var pred_test = data.hands_lms.toString();
    console.log(typeof pred_test);

    if(data.hands_lms==""){
      console.log("")
    }
    else if(pred_test.endsWith("Best of Luck ")){
      const regex = /<space>/g;
      text_data = text_data.replace(regex, " ");
      document.getElementById('transcribe').innerText = text_data;
      socket.close();
      if (mediaStream && mediaStream.getTracks) {
        mediaStream.getTracks().forEach(track => track.stop());
      }
    }
    else{
      document.getElementById('transcribe').innerText = data.hands_lms;
    }

  });

  // Stop the interval and video reading on close
  socket.addEventListener('close', function () {
    window.clearInterval(intervalId);
    video.pause();
  });

  return socket;
};

window.addEventListener('DOMContentLoaded', (event) => {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const cameraSelect = document.getElementById('camera-select');
  let socket;

  // List available cameras and fill select
  navigator.mediaDevices.enumerateDevices().then((devices) => {
    for (const device of devices) {
      if (device.kind === 'videoinput' && device.deviceId) {
        const deviceOption = document.createElement('option');
        deviceOption.value = device.deviceId;
        deviceOption.innerText = device.label;
        cameraSelect.appendChild(deviceOption);
      }
    }
  });

  // Start face detection on the selected camera on submit
  document.getElementById('form-connect').addEventListener('submit', (event) => {
    event.preventDefault();

    // Close previous socket is there is one
    if (socket) {
      socket.close();
    }

    const deviceId = cameraSelect.selectedOptions[0].value;
    socket = startHandDetection(video, canvas, deviceId);
  });

});