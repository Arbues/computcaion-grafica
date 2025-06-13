# Configuración del Entorno para los Proyectos de la Semana 14

Esta guía te ayudará a configurar un entorno de Python aislado para los proyectos de la Semana 14, utilizando las dependencias especificadas en `requirements.txt`. Seguir estos pasos es crucial para evitar conflictos entre las versiones de los paquetes.

## Prerrequisitos

*   **Python 3.8 - 3.10:** Asegúrate de tener instalada una versión compatible de Python. Puedes descargarlo desde [python.org](https://www.python.org/downloads/). (TensorFlow 2.12, especificado en tus requirements, tiene mejor compatibilidad con estas versiones de Python).

## Pasos para Configurar el Entorno

### 1. Crear un Entorno Virtual

Navega a tu carpeta `Semana 14` en la terminal y ejecuta el siguiente comando para crear un entorno virtual (lo llamaremos `.venv`):

```bash
cd ruta/a/tu/carpeta/Comp-Grafica-UNI/Semana 14
python -m venv .venv
```

### 2. Activar el Entorno Virtual

Una vez creado, necesitas activar el entorno:

*   **En Windows (cmd.exe):**
    ```bash
    .\.venv\Scripts\activate
    ```
*   **En Windows (PowerShell):**
    ```bash
    .\.venv\Scripts\Activate.ps1
    ```
    (Si encuentras un error de política de ejecución en PowerShell, puede que necesites ejecutar: `Set-ExecutionPolicy Unrestricted -Scope Process` en una PowerShell como Administrador, y luego intentar activar de nuevo).
*   **En macOS y Linux (bash/zsh):**
    ```bash
    source .venv/bin/activate
    ```

Sabrás que el entorno está activo porque verás `(.venv)` al principio de la línea de comandos de tu terminal.

### 3. Corregir el Archivo `requirements.txt` (Importante)

El archivo `requirements.txt` proporcionado tiene una especificación para `ml-dtypes==0.2.0` y `tensorflow==2.12.0`. La versión `2.12.0` de TensorFlow requiere `ml-dtypes` en una versión estrictamente menor a `0.2.0` (por ejemplo, `ml-dtypes>=0.1.0,<0.2.0`). Para evitar un conflicto de dependencias inmediato:

*   Abre tu archivo `c:\Users\kikhe\Documents\GitHub\Comp-Grafica-UNI\Semana 14\requirements.txt`.
*   Busca la línea:
    ```
    ml-dtypes==0.2.0
    ```
*   Cámbiala a:
    ```
    ml-dtypes==0.1.0
    ```
    Esta versión (`0.1.0`) es compatible tanto con `tensorflow==2.12.0` como con `jax==0.4.13` (que también está en tus requirements).

### 4. Instalar las Dependencias

Con el entorno virtual activado y el archivo `requirements.txt` corregido, instala todas las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

Esto puede tardar unos minutos ya que descargará e instalará todos los paquetes listados.

### 5. Verificar la Instalación (Opcional)

Puedes verificar si un paquete específico, como TensorFlow, se instaló correctamente:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```
Esto debería imprimir la versión de TensorFlow instalada (por ejemplo, `2.12.0`).

## Uso del Entorno

Cada vez que quieras trabajar en los proyectos de la Semana 14, asegúrate de activar el entorno virtual (`.\.venv\Scripts\activate` o `source .venv/bin/activate`) en tu terminal antes de ejecutar cualquier script de Python.

## Desactivar el Entorno Virtual

Cuando hayas terminado de trabajar, puedes desactivar el entorno virtual simplemente ejecutando:

```bash
deactivate
```

Esto te devolverá a tu intérprete de Python global.

---

Siguiendo estos pasos, deberías tener un entorno de trabajo limpio y funcional para tus proyectos, minimizando los problemas de compatibilidad de