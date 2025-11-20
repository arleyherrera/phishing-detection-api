# Iconos de la Extensión

Para que la extensión funcione correctamente, necesitas crear 3 iconos:

- `icon16.png` (16x16 píxeles)
- `icon48.png` (48x48 píxeles)
- `icon128.png` (128x128 píxeles)

## Opción 1: Usar un generador online

1. Ve a https://www.favicon-generator.org/
2. Sube una imagen de un escudo 🛡️ o candado 🔒
3. Descarga los tamaños 16x16, 48x48 y 128x128
4. Guárdalos en esta carpeta con los nombres correctos

## Opción 2: Usar emojis como placeholder

Puedes usar estos archivos temporales hasta que tengas iconos profesionales:

1. Descarga un icono de escudo desde https://emojipedia.org/shield/
2. Redimensiona a los tamaños necesarios
3. Guarda como PNG

## Opción 3: Crear con Python (Pillow)

```python
from PIL import Image, ImageDraw, ImageFont

def create_icon(size):
    # Crear imagen con fondo morado
    img = Image.new('RGB', (size, size), color='#667eea')
    draw = ImageDraw.Draw(img)

    # Dibujar escudo simple
    # ... código para dibujar

    img.save(f'icon{size}.png')

create_icon(16)
create_icon(48)
create_icon(128)
```

## Diseño recomendado

- **Color principal:** Morado (#667eea) o azul (#4f46e5)
- **Icono:** Escudo 🛡️ o candado 🔒
- **Fondo:** Gradiente o color sólido
- **Estilo:** Moderno, minimalista

## Temporalmente

Mientras no tengas iconos, la extensión usará el icono por defecto de Chrome.
La funcionalidad NO se ve afectada, solo la apariencia visual.
