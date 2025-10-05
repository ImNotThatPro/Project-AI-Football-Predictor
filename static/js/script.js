let result = null
async function getPrediction(){
    const teamA = document.getElementById('teamA').value;
    const teamB = document.getElementById('teamB').value;
    const url = `http://127.0.0.1:8000/predict?teamA=${encodeURIComponent(teamA)}&teamB=${encodeURIComponent(teamB)}`;

    document.getElementById('form-block').style.display ='none';
    document.getElementById('result-block').style.display ='block';
    

    const resultBox = document.getElementById('result');
    resultBox.innerText = 'Fetching data...'
    await new Promise(r => setTimeout(r, 800));
    resultBox.innerText = 'Running model prediction...';
    await new Promise(r => setTimeout(r, 1200));
    resultBox.innerText = 'Calculating win probabilities';

    const response = await fetch(url)
    result = await response.json();
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

//This is functionless for now, will see what does this even do
function showBetting(probabilities, teamA, teamB){
    document.getElementById('betting-block').style.display = 'block';
    
    const homeOdds = (1/ probabilities.HomeWin * 0.9).toFixed(2);
    const drawOdds = (1/ probabilities.Draw *0.9).toFixed(2);
    const awayOdds = (1/ probabilities.AwayWin * 0.9).toFixed(2);

    document.getElementById('homeTeamOdds').innerText = `HomeWin (${teamA}): ${homeOdds}x`
    document.getElementById('drawOdds').innerText = `Draw: ${drawOdds}x`
    document.getElementById('awayTeamOdds').innerText = `AwayWin (${teamB}): ${awayOdds}x`
    window.currentResult = result
}
let balance = 100
const balance_text = document.getElementById('balance')
balance_text.innerText = `Balance:${balance} `
//Some how this is working
document.getElementById('bet-result').addEventListener('click', () =>{
    if (result === null){
        document.getElementById('bet-result').innerText = '⚠ Please enter Home and Away team before placing an bet'
    }
    const betAmount = parseFloat(document.getElementById('betAmount').value);
    const choice = document.getElementById('betChoice').value;
    const oddsText = document.getElementById(choice === "HomeWin" ? "homeTeamOdds" : 
             choice === "Draw" ? "drawOdds" : "awayTeamOdds").innerText;
    const odds = parseFloat(oddsText.split(" ")[oddsText.split(" ").length-1].replace("x",""));
    
    if (isNaN(betAmount) || betAmount <=0){
        document.getElementById('bet-result').innerText= '⚠ Enter a valid bet amount.';
        return;
    };
    if (betAmount > balance){
        document.getElementById('bet-result').innerText = '⚠ Not enough balance to do this bet.';
        return;
    };
    if (choice === result.prediction) {
    const winnings = betAmount * odds;
    balance += winnings;
    document.getElementById("bet-result").innerText = `🎉 You WON! Balance: $${balance.toFixed(2)}`;
    } else {
        balance -= betAmount;
        document.getElementById("bet-result").innerText = `💔 You lost. Balance: $${balance.toFixed(2)}`;
    }
    document.getElementById("balance").innerText = `Balance: $${balance.toFixed(2)}`;
    console.log(balance)
}
)

document.addEventListener('DOMContentLoaded', ()=> {
    const tabs = document.querySelectorAll('.topbars, .topbars-active');
    const panels = document.querySelectorAll('.panel');

    tabs.forEach(t=>{
        t.addEventListener('click', () =>{
            tabs.forEach(x => x.classList.remove('active'));
            t.classList.add('active');
            panels.forEach(p => p.classList.remove('active'));
            const type = t.dataset.type;
            const panel = document.getElementById('panel-' +type)
            if (panel) panel.classList.add('active');
        });
    });
})
