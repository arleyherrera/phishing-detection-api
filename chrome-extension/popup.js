/**
 * Popup Script - Phishing Detector Extension
 *
 * Lógica principal de la extensión:
 * 1. Obtener URL de la pestaña activa
 * 2. Llamar a la API para predicción
 * 3. Mostrar resultado en la UI
 */

// Configuración de la API
const API_ENDPOINT = 'http://localhost:5000/predict';
// Para producción, cambiar a: 'https://tu-api.railway.app/predict'

// Elementos del DOM
const statusEl = document.getElementById('status');
const urlDisplayEl = document.getElementById('url-display');
const detailsEl = document.getElementById('details');
const predictionEl = document.getElementById('prediction');
const confidenceEl = document.getElementById('confidence');
const riskLevelEl = document.getElementById('risk-level');
const featuresEl = document.getElementById('features');
const warningEl = document.getElementById('warning');

/**
 * Inicializar la extensión cuando se abre el popup
 */
document.addEventListener('DOMContentLoaded', async () => {
  console.log('Phishing Detector iniciado');

  try {
    // Obtener la pestaña activa
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab || !tab.url) {
      showError('No se pudo obtener la URL de la pestaña activa');
      return;
    }

    const url = tab.url;
    console.log('URL a analizar:', url);

    // Validar que sea una URL analizable
    if (isSpecialURL(url)) {
      showSpecialURLMessage(url);
      return;
    }

    // Mostrar URL en la UI
    urlDisplayEl.textContent = truncateURL(url, 60);

    // Llamar a la API
    await analyzeURL(url);

  } catch (error) {
    console.error('Error en DOMContentLoaded:', error);
    showError('Error al inicializar la extensión: ' + error.message);
  }
});

/**
 * Verificar si es una URL especial que no se puede analizar
 */
function isSpecialURL(url) {
  return url.startsWith('chrome://') ||
         url.startsWith('chrome-extension://') ||
         url.startsWith('edge://') ||
         url.startsWith('about:') ||
         url.startsWith('file://');
}

/**
 * Mostrar mensaje para URLs especiales
 */
function showSpecialURLMessage(url) {
  statusEl.className = 'status error';
  statusEl.innerHTML = `
    <div class="icon">⚠️</div>
    <div>No se puede analizar</div>
  `;

  urlDisplayEl.textContent = truncateURL(url, 60);

  warningEl.style.display = 'block';
  warningEl.textContent = 'Las páginas internas del navegador (chrome://, file://, etc.) no pueden ser analizadas.';
}

/**
 * Analizar URL llamando a la API
 */
async function analyzeURL(url) {
  try {
    console.log('Llamando a la API:', API_ENDPOINT);

    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: url })
    });

    console.log('Respuesta HTTP:', response.status);

    if (!response.ok) {
      throw new Error(`Error del servidor: ${response.status}`);
    }

    const data = await response.json();
    console.log('Datos recibidos:', data);

    // Mostrar resultado
    displayResult(data);

  } catch (error) {
    console.error('Error al analizar URL:', error);

    // Verificar si es un error de red
    if (error.message.includes('Failed to fetch')) {
      showError('No se pudo conectar con la API. Asegúrate de que el servidor esté ejecutándose en ' + API_ENDPOINT);
    } else {
      showError('Error al analizar la URL: ' + error.message);
    }
  }
}

/**
 * Mostrar el resultado de la predicción
 */
function displayResult(data) {
  const isSafe = data.prediction === 'safe';
  const confidence = (data.confidence * 100).toFixed(1);

  // Actualizar estado
  statusEl.className = isSafe ? 'status safe' : 'status danger';
  statusEl.innerHTML = `
    <div class="icon">${isSafe ? '✅' : '🚨'}</div>
    <div>${isSafe ? 'Sitio Seguro' : '¡PELIGRO! Sitio de Phishing'}</div>
  `;

  // Mostrar detalles
  detailsEl.classList.add('show');

  predictionEl.textContent = isSafe ? 'Legítimo' : 'Phishing';
  predictionEl.style.color = isSafe ? '#16a34a' : '#dc2626';

  confidenceEl.textContent = confidence + '%';

  // Nivel de riesgo basado en la confianza
  const riskLevel = getRiskLevel(data.prediction, data.confidence);
  riskLevelEl.textContent = riskLevel.text;
  riskLevelEl.style.color = riskLevel.color;

  featuresEl.textContent = `${data.features_extracted}/${data.features_required}`;

  // Mostrar advertencia si hay
  if (data.warning) {
    warningEl.style.display = 'block';
    warningEl.textContent = data.warning;
  }

  // Advertencia adicional si es phishing
  if (!isSafe) {
    warningEl.style.display = 'block';
    warningEl.innerHTML = `
      <strong>⚠️ ADVERTENCIA:</strong><br>
      Este sitio ha sido identificado como phishing con ${confidence}% de confianza.
      <br><br>
      <strong>NO ingreses:</strong>
      <ul style="margin: 5px 0 0 20px; font-size: 11px;">
        <li>Contraseñas</li>
        <li>Datos bancarios</li>
        <li>Información personal</li>
      </ul>
    `;
  }
}

/**
 * Calcular nivel de riesgo
 */
function getRiskLevel(prediction, confidence) {
  if (prediction === 'safe') {
    return {
      text: 'BAJO',
      color: '#16a34a'
    };
  } else {
    if (confidence > 0.9) {
      return {
        text: 'CRÍTICO',
        color: '#dc2626'
      };
    } else if (confidence > 0.7) {
      return {
        text: 'ALTO',
        color: '#ea580c'
      };
    } else {
      return {
        text: 'MEDIO',
        color: '#f59e0b'
      };
    }
  }
}

/**
 * Mostrar mensaje de error
 */
function showError(message) {
  statusEl.className = 'status error';
  statusEl.innerHTML = `
    <div class="icon">❌</div>
    <div>Error de Análisis</div>
  `;

  warningEl.style.display = 'block';
  warningEl.textContent = message;

  console.error(message);
}

/**
 * Truncar URL para mostrar
 */
function truncateURL(url, maxLength) {
  if (url.length <= maxLength) {
    return url;
  }
  return url.substring(0, maxLength - 3) + '...';
}

/**
 * Formatear timestamp
 */
function formatTime() {
  const now = new Date();
  return now.toLocaleTimeString('es-PE', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
}
