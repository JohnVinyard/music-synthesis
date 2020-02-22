const Audio = Vue.component('audio-player', {
  template: '#audio-template',
  props: ['source'],
});

const Image = Vue.component('image-viewer', {
  template: '#image-template',
  props: ['source']
});

const app = new Vue({
  el: '#app',
  data: { reportData: null }
});

fetch('report_data.json')
  .then((response) => {
    return response.json();
  })
  .then((data) => {
    console.log(data);
    app.reportData = data;
  });
