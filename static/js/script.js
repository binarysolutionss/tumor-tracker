document.addEventListener('DOMContentLoaded', function(){
    const form = document.querySelector('.card-form');
    const fileInput = document.getElementById('select-image');
    const submitBtn = document.querySelector('.submit-btn');

    fileInput.addEventListener('change', function(e){
        const file = e.target.files[0];
        if (file){
            //Validation of image data type
            if (!file.type.startsWith('image/')){
                displayError('Please selectan image file.');
                fileInput.value='';
                return;
            }

            //Validation of file size (max:5mb)
            if (file.size> 5 * 1024 * 1024){
                displayError('File size must be less than 5MB.');
                fileInput.value='';
                return;
            }
        }
    });

    form.addEventListener('submit', async function(e){
        e.preventDefault();

        if (!fileInput.files[0]){
            displayError('Please select an image file.');
            return;
        }

        const formData = new formData(form);
        submitBtn.disabled=True;
        submitBtn.value='Analyzing...';

        try{
            const response=await fetch('/0', {
                method:'POST',
                body:formData
            });

            const html = await response.text();
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const newResult = doc.getElementById('result');

            document.getElementById('result').innerHTML = newResult.innerHTML;
        }
        catch (error){
            displayError('An error has occured whil processing the image.');
        }
        finally{
            submitBtn.disabled=false;
            submitBtn.value="Upload";
        }
    });

    function displayError(message){
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `<div class="error" style="color:red; padding:1rem;">${message}</div>`;
    }
});