# MLOps Pipeline con GitHub Actions

Este repositorio ha sido desarrollado por: **Paula Camprecios**

Este repositorio implementa una solución de MLOps con **control de calidad en los pull requests** y **entrenamiento automático de modelos con seguimiento en MLflow**. A través de diferentes GitHub Actions, aseguramos que el código subido esté testeado, validado y correctamente integrado antes de desplegar un modelo entrenado.

## Estructura del repositorio

Para entender el repositorio, se lista a bajo, las carpetas y elementos básicos de cada uno:

├── .github/workflows/
│ ├── integration.yml # Ejecuta tests al crear un Pull Request
│ └── build.yml # Entrena y registra el modelo al hacer push a main
├── data/ # (Ignorado) Contendrá los datasets necesarios
├── src/ # Código fuente principal
│ ├── main.py # Script principal de la aplicación
│ └── data_loader.py # Funciones para cargar y preprocesar los datos
├── scripts/
│ └── registry_model.py # Encargado de registrar modelos entrenados
├── unit_tests/ # Tests unitarios de las funciones en src/
├── requirements.txt # Dependencias del proyecto
├── .gitignore # Archivos y carpetas ignoradas por git


---

## GitHub Actions

El repositorio está automatizado con **GitHub Actions** para asegurar calidad y reproducibilidad.

### `integration.yml` – Validación de Pull Requests

Ubicado en `.github/workflows/integration.yml`, este workflow se activa cuando se abre un Pull Request.

**Pasos:**

1. Checkout del código
2. Setup de Python (v3.10)
3. Instalación de dependencias (`requirements.txt`)
4. Ejecución de tests (`unit_tests/`), que usan funciones de `src/`
5. Registro de resultados en los comentarios del Pull Request
6. Si hay errores, el PR es marcado como fallido

Todo Pull Request requiere al menos un revisor antes de ser integrado a `main`.

---

### `build.yml` – Entrenamiento y Registro del Modelo

Este workflow se ejecuta al hacer **push a la rama `main`**.

**Pasos:**

1. Define un nombre para el run
2. Checkout del código
3. Instala Python y dependencias
4. Descarga los datasets (sin subirlos al repositorio)
5. Entrena y guarda el modelo con un ID específico
6. Testea el modelo entrenado
7. Registra automáticamente el modelo en **MLflow**

Este workflow usa **secrets y variables de entorno configuradas en GitHub**, evitando exponer datos sensibles.

## Tests y Calidad

La carpeta `unit_tests/` contiene tests unitarios que prueban las funciones implementadas en `src/`.  
Estos se ejecutan automáticamente en cada Pull Request mediante `integration.yml`.

## Scripts

La carpeta `scripts/` incluye herramientas auxiliares como:

- `registry_model.py`: Encargado de registrar modelos entrenados en MLflow.





