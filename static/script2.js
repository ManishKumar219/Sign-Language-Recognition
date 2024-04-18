const IMAGE_INTERVAL_MS = 300;
let count = 0;
console.log("Succes");
console.log(IMAGE_INTERVAL_MS)

let info = document.getElementById('info')
let result = document.getElementById('result')


const startHandDetection = (video, canvas, deviceId) => {
  const socket = new WebSocket('ws://localhost:8000/calc-mode');
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


  // script.js:59 Uncaught ReferenceError: pred_test is not defined
  // at WebSocket.<anonymous> (script.js:59:17)




  // Listen for messages
  socket.addEventListener('message', function (event) {
    // drawFaceRectangles(video, canvas, JSON.parse(event.data));
    var data = JSON.parse(event.data);
    console.log(count++);
    
    // var text_data = document.getElementById('transcribe').innerText;
    var recv_result = data.hands_lms.toString();
    var recv_info = data.info.toString();

    console.log(recv_info);
    console.log(data.hands_lms);

      result.innerText = data.hands_lms;
      info.innerText = data.info;

      if(recv_info == "Result"){
        socket.close();
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