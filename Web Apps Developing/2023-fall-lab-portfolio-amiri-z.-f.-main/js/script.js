document.addEventListener('DOMContentLoaded', function () {
  const videoSection = document.getElementById('video-section')
  if (videoSection != null) {
    videoSection.innerHTML = '<p><b>The best goal ever! It was better than everything!</b></p> <iframe width="560" height="315" src="https://www.youtube.com/embed/ZC_XihZGPEg?si=KAz7HItciZuP2wU3" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>'
  }
  document.getElementById('toggle-menu').addEventListener('click', function () {
    const nav = document.querySelector('nav')
    nav.classList.toggle('show-menu')
  })

  document
    .getElementById('toggle-dark-mode')
    .addEventListener('click', function () {
      const body = document.body
      body.classList.toggle('dark-mode')
    })

  document
    .getElementById('toggle-dark-mode-nav')
    .addEventListener('click', function () {
      const body = document.body
      body.classList.toggle('dark-mode')
    })

  document.addEventListener('click', function (event) {
    const nav = document.querySelector('nav')
    if (!event.target.closest('nav') && event.target.id !== 'toggle-menu') {
      nav.classList.remove('show-menu')
    }
  })
})
