window.onresize = function (){
    if (window.outerWidth < 400 || window.outerHeight < 300){
        let newwidth = max(window.outerWidth, 400)
        let newheight = max(window.outerHeight, 300)
        window.resizeTo(newwidth, newheight);
    }
}




document.querySelector(".translate").onclick = function () {  
    eel.translate(document.getElementById('org').value)(function(string){                      
      document.getElementById('trans').value = string;
    })
  }