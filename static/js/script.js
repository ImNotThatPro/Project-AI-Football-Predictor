async function getPrediction(){
    const teamA = document.getElementById('teamA').value
    const teamB = document.getElementById('teamB').value
    const url = `http://127.0.0.1:8000/predict?teamA=${encodeURIComponent(teamA)}&teamB=${encodeURIComponent(teamB)}`;
    const response = await fetch(url)
    const result = await response.json();
    console.log(result)
    console.log(url)
    document.getElementById('result').innerText = `Home team: ${teamA} \n Away team: ${teamB} \n Prediction: ${result.prediction} \n Accuracy ~ 52%`
}

