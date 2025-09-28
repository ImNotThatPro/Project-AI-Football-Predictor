async function getPrediction(){
    document.getElementById('form-block').style.display ='none';
    document.getElementById('result-block').style.display ='block';
    const teamA = document.getElementById('teamA').value;
    const teamB = document.getElementById('teamB').value;

    const resultBox = document.getElementById('result');
    resultBox.innerText = 'Fetching data...'
    await new Promise(r => setTimeout(r, 800));
    resultBox.innerText = 'Running model prediction...';
    await new Promise(r => setTimeout(r, 1200));
    resultBox.innerText = 'Calculating win probabilities';

    const url = `http://127.0.0.1:8000/predict?teamA=${encodeURIComponent(teamA)}&teamB=${encodeURIComponent(teamB)}`;
    const response = await fetch(url)
    const result = await response.json();
    console.log(result);
    console.log(url);
    await new Promise(r => setTimeout(r, 1000));
    document.getElementById('result').innerText = `\n Prediction: ${result.prediction} \n
                                                 Home team: ${teamA} Win chance : ${(result.probs['HomeWin'] * 100).toFixed(1)} % \n
                                                 Away team: ${teamB} Win chance : ${(result.probs['AwayWin'] * 100).toFixed(1)} % \n
                                                 Draw chance : ${(result.probs['Draw'] * 100).toFixed(1)} % \n
                                                 Model Accuracy ~ 52%` 
    

    console.log(result.probs);
}

async function goBack(){
    document.getElementById('form-block').style.display ='block';
    document.getElementById('result-block').style.display ='none';
}

