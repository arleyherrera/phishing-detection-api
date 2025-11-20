# Phishing Detector - Extensión de Chrome

Extensión de Google Chrome que detecta sitios de phishing en tiempo real usando Machine Learning.

## 🚀 Instalación

### Paso 1: Cargar la extensión en Chrome

1. Abre Chrome y navega a `chrome://extensions/`
2. Activa el **Modo de desarrollador** (toggle en la esquina superior derecha)
3. Haz clic en **Cargar extensión sin empaquetar**
4. Selecciona la carpeta `chrome-extension`
5. La extensión debería aparecer en tu lista de extensiones

### Paso 2: Asegurar que la API esté ejecutándose

La extensión necesita que la API esté corriendo en `http://localhost:5000`:

```bash
# Desde la carpeta raíz del proyecto
python api_phishing.py
```

La API debe mostrar:
```
================================================================================
     API de Deteccion de Phishing - Para Extension Chrome
================================================================================

Modelo: Random Forest RAPIDO (99.95% accuracy)
Features requeridas: 56

 * Running on http://localhost:5000
```

## 📖 Uso

1. Navega a cualquier sitio web
2. Haz clic en el icono de la extensión (🛡️) en la barra de herramientas
3. La extensión analizará automáticamente la URL actual
4. Verás el resultado en segundos:
   - ✅ **Sitio Seguro** (verde) - sitio legítimo
   - 🚨 **¡PELIGRO!** (rojo) - sitio de phishing detectado

## 🔧 Configuración para Producción

Para usar la API desplegada en Railway (en lugar de localhost):

1. Abre `popup.js`
2. Cambia la línea 11:
   ```javascript
   // Desarrollo
   const API_ENDPOINT = 'http://localhost:5000/predict';

   // Producción
   const API_ENDPOINT = 'https://tu-api.railway.app/predict';
   ```
3. Actualiza `manifest.json` para incluir el dominio de Railway en `host_permissions`

## ⚠️ Limitaciones

La extensión **NO puede analizar**:
- Páginas internas de Chrome (`chrome://...`)
- Archivos locales (`file://...`)
- Otras extensiones (`chrome-extension://...`)

## 🛠️ Desarrollo

### Estructura de archivos

```
chrome-extension/
├── manifest.json       # Configuración de la extensión
├── popup.html          # Interfaz de usuario
├── popup.js            # Lógica de la extensión
├── icons/              # Iconos (16x16, 48x48, 128x128)
└── README.md           # Este archivo
```

### Debugging

Para ver los logs de la extensión:
1. Abre la extensión
2. Click derecho en el popup → **Inspeccionar**
3. Ve a la pestaña **Console**

## 📊 Modelo de Machine Learning

La extensión usa un modelo Random Forest entrenado con:
- **48,938 URLs** (phishing + legítimas)
- **56 features** extraídas de cada URL
- **99.95% de precisión**
- **<100ms** de tiempo de respuesta

## 🔐 Privacidad

- ✅ Solo analiza la URL cuando haces clic en el icono (permiso `activeTab`)
- ✅ No registra historial de navegación
- ✅ No almacena datos personales
- ✅ No requiere permisos invasivos

## 🐛 Troubleshooting

### Error: "No se pudo conectar con la API"

**Solución:** Verifica que la API esté ejecutándose:
```bash
python api_phishing.py
```

Debería responder en `http://localhost:5000`

### Error: "No se puede analizar"

**Causa:** Estás en una página especial (chrome://, file://, etc.)

**Solución:** Navega a un sitio web normal (ej: https://www.google.com)

### La extensión no aparece en la barra

**Solución:**
1. Ve a `chrome://extensions/`
2. Encuentra "Phishing Detector"
3. Haz clic en el ícono de pin 📌

## 📝 Licencia

Este proyecto es parte de una tesis de grado.
