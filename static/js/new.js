//Smooth scroll for navigation link
document.querySelectorAll('a[href^="#').forEach(anchor=>{
    anchor.addEventListener('click',function(e){
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior:'smooth'
        });
    });
});
//End of Smooth scroll for navigation link

// image upload
document.getElementById('select-image').addEventListener('change', async(e)=>{
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append('image', file);

    try{
        const response = await fetch('/predict', {
            method:'POST',
            body:formData
        });
        const result = await response.json();

        if (result.error){
            displayError(result.error);
        }
        else{
            displayResult(result);
        }
    }
    catch (error){
        displayError(error.message);
    }
});

function displayResult(result){
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML=`
        <div id="result ${result.result}">
            <h3>Detection Result:</div>
            <div class="confidence">
                Confidence: ${result.confidence}
            </div>
            <p>${result.info.description}</p>
            <div class="recommendations">
                <h4>Recommenndations:</h4>
                <ul>
                    ${result.info,recommendations.map(rec=>`<li>${rec}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
}

function displayError(error){
    const resultDiv=document.getElementById('result');
    resultDiv.innerHTML=`<div class="error">Error: ${error}</div>`;
}