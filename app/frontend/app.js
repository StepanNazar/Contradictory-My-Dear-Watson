function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].classList.remove("active");
    }
    tablinks = document.getElementsByClassName("tab-btn");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].classList.remove("active");
    }
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

document.getElementById('predict-btn').addEventListener('click', async () => {
    const premise = document.getElementById('premise').value;
    const hypothesis = document.getElementById('hypothesis').value;
    const language = document.getElementById('lang-select').value;
    const resultCard = document.getElementById('prediction-result');
    const labelSpan = document.getElementById('label-result');
    const probsContainer = document.getElementById('probs-container');

    if (!premise || !hypothesis) {
        alert("Please enter both premise and hypothesis.");
        return;
    }

    // Hide previous result and show loading state if desired (optional)
    resultCard.classList.add('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ premise, hypothesis, language })
        });

        const data = await response.json();
        
        labelSpan.innerText = data.label_name.toUpperCase();
        labelSpan.style.color = data.label_name === 'contradiction' ? '#c53030' : 
                               data.label_name === 'entailment' ? '#2f855a' : '#2c3e50';
        
        probsContainer.innerHTML = '';
        if (data.probabilities) {
            Object.entries(data.probabilities).forEach(([name, prob]) => {
                const wrapper = document.createElement('div');
                wrapper.className = 'prob-bar-wrapper';
                wrapper.innerHTML = `
                    <div class="prob-label"><span>${name}</span><span>${(prob * 100).toFixed(1)}%</span></div>
                    <div class="prob-bar"><div class="prob-fill" style="width: ${prob * 100}%"></div></div>
                `;
                probsContainer.appendChild(wrapper);
            });
        }
        
        resultCard.classList.remove('hidden');
    } catch (err) {
        alert("Error: " + err.message);
    }
});

document.getElementById('detect-btn').addEventListener('click', async () => {
    const text = document.getElementById('bulk-text').value;
    const language = document.getElementById('lang-batch').value;
    const resultCard = document.getElementById('contradiction-result');
    const matchesList = document.getElementById('matches-list');

    if (!text) {
        alert("Please enter some text.");
        return;
    }

    resultCard.classList.add('hidden');

    try {
        const response = await fetch('/detect-contradictions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, language })
        });

        const data = await response.json();
        
        matchesList.innerHTML = '';
        
        if (data.contradictions.length === 0) {
            matchesList.innerHTML = '<p>No contradictions found.</p>';
        } else {
            data.contradictions.forEach(m => {
                const div = document.createElement('div');
                div.className = 'contradiction-item';
                div.innerHTML = `
                    <p><strong>Statement A:</strong> ${m.premise}</p>
                    <p><strong>Statement B:</strong> ${m.hypothesis}</p>
                    <p><small>Confidence: ${(m.probability * 100).toFixed(1)}%</small></p>
                `;
                matchesList.appendChild(div);
            });
        }
        
        resultCard.classList.remove('hidden');
    } catch (err) {
        alert("Error: " + err.message);
    }
});
