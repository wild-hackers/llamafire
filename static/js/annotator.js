document.addEventListener('DOMContentLoaded', function() {
    const processBtn = document.getElementById('processBtn');
    const refreshBtn = document.getElementById('refreshBtn');
    const progressBar = document.querySelector('.progress-bar');
    const progressDiv = document.getElementById('progress');
    const progressText = document.getElementById('progress-text');

    processBtn.addEventListener('click', function() {
        console.log('Button clicked');
        console.log('Disabled attribute:', processBtn.hasAttribute('disabled'));
        console.log('Button classes:', processBtn.className);
        
        // Check if button is disabled (all images processed)
        if (processBtn.disabled || processBtn.hasAttribute('disabled')) {
            console.log('Button is disabled - showing alert');
            alert('All images have already been processed!');
            return;
        }

        console.log('Button is enabled - starting processing');
        processBtn.disabled = true;
        progressDiv.style.display = 'block';
        
        fetch('/process', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'skipped') {
                alert(data.message);
                processBtn.disabled = true;
                processBtn.classList.remove('btn-primary');
                processBtn.classList.add('btn-secondary');
            } else if (data.status === 'success') {
                alert('Processing completed successfully!');
                location.reload();
            } else {
                alert('Error during processing: ' + data.error);
            }
        })
        .catch(error => {
            alert('Error: ' + error);
        })
        .finally(() => {
            progressDiv.style.display = 'none';
        });
    });

    refreshBtn.addEventListener('click', function() {
        location.reload();
    });

    // Set up SSE for real-time progress updates
    const evtSource = new EventSource('/stream');
    evtSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.is_processing) {
            progressDiv.style.display = 'block';
            const percent = (data.processed / data.total) * 100;
            progressBar.style.width = percent + '%';
            progressText.textContent = `Processing ${data.current_image} (${data.processed}/${data.total})`;
        } else {
            progressDiv.style.display = 'none';
        }
    };
}); 