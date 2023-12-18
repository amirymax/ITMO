let autoSwitch = false;
let intervalId;
let audioPlayer = document.getElementById('audio-player');
let currentSlide = 0;
let audioVisualization = document.getElementById('audio-visualization');
let audioContext, analyser, bufferLength, dataArray;

function setupAudio() {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    let source = audioContext.createMediaElementSource(audioPlayer);
    source.connect(analyser);
    analyser.connect(audioContext.destination);
    analyser.fftSize = 256;
    bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);
}

function visualize() {
    analyser.getByteFrequencyData(dataArray);

    let sum = dataArray.reduce((a, b) => a + b, 0);
    let average = sum / bufferLength;

    audioVisualization.style.height = average * 2 + 'px';
}

function toggleVisualization() {
    audioVisualization.style.display = audioVisualization.style.display === 'none' ? 'block' : 'none';
}

function showSlide(index) {
    const slides = document.querySelectorAll('.slide');
    slides.forEach((slide, i) => {
        slide.style.display = i === index ? 'block' : 'none';
    });
}

function showText(index) {
    const textElements = document.querySelectorAll('.image-text');
    textElements.forEach((text, i) => {
        text.style.display = i === index ? 'block' : 'none';
    });
}

function showComment(comment) {
    console.log(comment);
}

function nextImage() {
    currentSlide = (currentSlide + 1) % 3;
    showSlide(currentSlide);
    showText(currentSlide);
    applyAnimation();
}

function applyAnimation() {
    const gallery = document.getElementById('gallery');
    gallery.classList.add('button-gradient');
    setTimeout(() => {
        gallery.classList.remove('button-gradient');
    }, 500);
}

function prevImage() {
    currentSlide = (currentSlide - 1 + 3) % 3;
    showSlide(currentSlide);
    showText(currentSlide);
}

function toggleAuto() {
    autoSwitch = !autoSwitch;
    if (autoSwitch) {
        intervalId = setInterval(nextImage, getTiming() * 1000);
    } else {
        clearInterval(intervalId);
    }
}

function getTiming() {
    return parseInt(document.getElementById('timing').value, 10);
}

function setTiming() {
    if (autoSwitch) {
        clearInterval(intervalId);
        intervalId = setInterval(nextImage, getTiming() * 1000);
    }
}

function toggleSound() {
    if (audioPlayer.paused) {
        const selectedAudio = document.getElementById('audio');
        audioPlayer.src = selectedAudio.value;
        audioPlayer.play();
        setupAudio();
        toggleVisualization();
        visualize();
    } else {
        audioPlayer.pause();
        toggleVisualization();
    }
}

function toggleTheme() {
    const body = document.body;
    body.classList.toggle('dark-theme');

    const themeSwitch = document.getElementById('theme-switch');
    themeSwitch.innerText = body.classList.contains('dark-theme') ? 'Светлая тема' : 'Темная тема';
}

document.addEventListener('DOMContentLoaded', function () {
    showSlide(currentSlide);
    showText(currentSlide);
});
