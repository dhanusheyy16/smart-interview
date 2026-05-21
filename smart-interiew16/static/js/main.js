document.addEventListener("DOMContentLoaded", () => {
    // --- Elements ---
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");
    const openWebcamBtn = document.getElementById("open-webcam-btn");
    
    const webcamSection = document.getElementById("webcam-section");
    const videoUploadSection = document.getElementById("video-upload-section");
    
    const webcamVideo = document.getElementById("webcam-video");
    const startRecordBtn = document.getElementById("start-record-btn");
    const stopRecordBtn = document.getElementById("stop-record-btn");
    const cancelWebcamBtn = document.getElementById("cancel-webcam-btn");
    
    const loadingOverlay = document.getElementById("loading-overlay");
    const loadingText = document.getElementById("loading-text");
    const step1 = document.getElementById("step-1");
    const step2 = document.getElementById("step-2");
    const step3 = document.getElementById("step-3");
    const step4 = document.getElementById("step-4");
    const step5 = document.getElementById("step-5");
    
    const toast = document.getElementById("toast");

    // --- State variables ---
    let mediaStream = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let recordTimer = null;
    let recordSeconds = 0;

    // --- Helper: Show Toast notifications ---
    function showToast(message, duration = 4000) {
        toast.textContent = message;
        toast.style.display = "block";
        setTimeout(() => {
            toast.style.display = "none";
        }, duration);
    }

    // --- Drag and Drop File Handlers ---
    if (uploadArea) {
        uploadArea.addEventListener("click", () => fileInput.click());

        ["dragenter", "dragover"].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                uploadArea.classList.add("dragover");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                uploadArea.classList.remove("dragover");
            }, false);
        });

        uploadArea.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleUploadedFile(files[0]);
            }
        });

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                handleUploadedFile(e.target.files[0]);
            }
        });
    }

    function handleUploadedFile(file) {
        // Validate file type is video
        if (!file.type.startsWith("video/")) {
            showToast("⚠️ Invalid file! Please upload a video file.");
            return;
        }

        // Validate file size is under 64MB
        if (file.size > 64 * 1024 * 1024) {
            showToast("⚠️ File is too large! Maximum limit is 64MB.");
            return;
        }

        uploadFileToServer(file);
    }

    // --- Webcam Capture Controllers ---
    if (openWebcamBtn) {
        openWebcamBtn.addEventListener("click", async () => {
            try {
                logger("Requesting camera/microphone access...");
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, frameRate: { ideal: 30 } },
                    audio: true
                });
                
                webcamVideo.srcObject = mediaStream;
                webcamVideo.muted = true; // Avoid feedback squeal locally
                
                // Hide file upload, show camera recorder pane
                videoUploadSection.style.display = "none";
                webcamSection.style.display = "block";
                
                startRecordBtn.style.display = "inline-flex";
                stopRecordBtn.style.display = "none";
                recordedChunks = [];
            } catch (err) {
                console.error("Camera access denied:", err);
                showToast("❌ Camera/Microphone access denied. Please allow permissions in your browser.");
            }
        });
    }

    if (cancelWebcamBtn) {
        cancelWebcamBtn.addEventListener("click", () => {
            stopWebcamStream();
            webcamSection.style.display = "none";
            videoUploadSection.style.display = "block";
        });
    }

    function stopWebcamStream() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        webcamVideo.srcObject = null;
        clearInterval(recordTimer);
        recordSeconds = 0;
    }

    if (startRecordBtn) {
        startRecordBtn.addEventListener("click", () => {
            if (!mediaStream) return;
            
            recordedChunks = [];
            let options = { mimeType: 'video/webm;codecs=vp9,opus' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = { mimeType: 'video/webm;codecs=vp8,opus' };
            }
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = { mimeType: 'video/webm' };
            }

            try {
                mediaRecorder = new MediaRecorder(mediaStream, options);
            } catch (e) {
                // Fallback to standard
                mediaRecorder = new MediaRecorder(mediaStream);
            }

            mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const recordedBlob = new Blob(recordedChunks, { type: "video/webm" });
                stopWebcamStream();
                
                // Directly upload the webcam recorded video blob
                const file = new File([recordedBlob], "webcam_interview.webm", { type: "video/webm" });
                uploadFileToServer(file);
            };

            mediaRecorder.start(1000); // chunk every 1 second
            
            // Set up Recording UI State
            startRecordBtn.style.display = "none";
            stopRecordBtn.style.display = "inline-flex";
            
            // Start recording timer
            recordSeconds = 0;
            updateRecordTimerDisplay();
            recordTimer = setInterval(() => {
                recordSeconds++;
                updateRecordTimerDisplay();
                // Max recording duration: 90 seconds
                if (recordSeconds >= 90) {
                    showToast("⏰ Maximum time limit reached (90s). Processing video...");
                    stopRecordBtn.click();
                }
            }, 1000);
            
            showToast("🔴 Recording started. Speak clearly!");
        });
    }

    if (stopRecordBtn) {
        stopRecordBtn.addEventListener("click", () => {
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
            }
        });
    }

    function updateRecordTimerDisplay() {
        const mins = Math.floor(recordSeconds / 60);
        const secs = recordSeconds % 60;
        const timerText = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        document.querySelector(".timer").textContent = timerText;
    }

    // --- Server Upload & Status Polling ---
    function uploadFileToServer(file) {
        // Show loaders
        loadingOverlay.classList.add("active");
        resetLoadingSteps();
        
        const formData = new FormData();
        formData.append("video", file);

        // Run an intelligent status interval on client side to simulate high tech analysis phases
        animateAnalysisStatus();

        fetch("/analyze", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw new Error(errData.error || "Server processing failed.");
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                markStepCompleted(step5);
                loadingText.textContent = "Redirecting to your analysis dashboard...";
                setTimeout(() => {
                    window.location.href = data.redirect_url;
                }, 1000);
            } else {
                throw new Error(data.error || "Analysis failed.");
            }
        })
        .catch(error => {
            console.error("Analysis Pipeline crashed:", error);
            loadingOverlay.classList.remove("active");
            showToast(`❌ Error: ${error.message}`);
        });
    }

    // --- Loading UI Animations ---
    function resetLoadingSteps() {
        [step1, step2, step3, step4, step5].forEach(step => {
            step.className = "progress-step";
        });
    }

    function markStepActive(stepElement) {
        stepElement.className = "progress-step active";
    }

    function markStepCompleted(stepElement) {
        stepElement.className = "progress-step completed";
        const dot = stepElement.querySelector(".step-dot");
        if (dot) dot.innerHTML = "✓";
    }

    function animateAnalysisStatus() {
        // Step 1: Upload & Audio Extract (0 - 3s)
        markStepActive(step1);
        loadingText.textContent = "Isolating audio frequencies from video container...";

        setTimeout(() => {
            markStepCompleted(step1);
            markStepActive(step2);
            loadingText.textContent = "Transcribing speaking segments using Whisper Speech AI...";
        }, 3000);

        // Step 2: Transcribe (3 - 7s)
        setTimeout(() => {
            markStepCompleted(step2);
            markStepActive(step3);
            loadingText.textContent = "Calculating pitch, pace, pauses, and verbal fillers...";
        }, 7000);

        // Step 3: Audio Analysis (7 - 11s)
        setTimeout(() => {
            markStepCompleted(step3);
            markStepActive(step4);
            loadingText.textContent = "Analyzing posture, smiles, and facial coordinate stability...";
        }, 11000);

        // Step 4: Video Analysis (11 - 15s)
        setTimeout(() => {
            markStepCompleted(step4);
            markStepActive(step5);
            loadingText.textContent = "Connecting to Azure OpenAI for expert feedback coaching...";
        }, 15000);
    }

    function logger(msg) {
        console.log(`[SmartInterview] ${msg}`);
    }
});
