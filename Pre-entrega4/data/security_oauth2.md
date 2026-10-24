---
title: Seguridad con OAuth2 y JWT en FastAPI
category: seguridad
tags: oauth2, jwt, oauth2passwordbearer, scopes, cors, hashing
---

# Seguridad con OAuth2 y JWT en FastAPI

FastAPI ofrece herramientas en el módulo `fastapi.security` para implementar autenticación y autorización siguiendo estándares abiertos. Como todas estas herramientas se integran con OpenAPI, la documentación interactiva en `/docs` muestra un botón **Authorize** que permite probar los endpoints protegidos desde el navegador.

## El flujo password de OAuth2

El flujo más común en APIs propias es el flujo *password* de OAuth2. El cliente envía usuario y contraseña a un endpoint de token (convencionalmente `/token`) como datos de formulario, no como JSON. El servidor valida las credenciales y devuelve un *access token*. A partir de ahí, el cliente envía ese token en el header `Authorization: Bearer <token>` en cada petición.

Para recibir el formulario se usa `OAuth2PasswordRequestForm` como dependencia: `form_data: Annotated[OAuth2PasswordRequestForm, Depends()]`. Este formulario expone `form_data.username`, `form_data.password` y `form_data.scopes`. Como se trata de datos de formulario, es necesario tener instalado el paquete `python-multipart`; si falta, FastAPI lanza un error al iniciar indicando que no puede procesar formularios.

## OAuth2PasswordBearer

La clase **`OAuth2PasswordBearer`** declara el esquema de seguridad: `oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")`. El parámetro `tokenUrl` es relativo e indica a la documentación dónde obtener el token. Al usarla como dependencia (`token: Annotated[str, Depends(oauth2_scheme)]`), FastAPI busca el header `Authorization`, verifica que tenga el formato `Bearer` y entrega el token como string. Si el header falta o es inválido, responde automáticamente con **401 Unauthorized** e incluye el header `WWW-Authenticate: Bearer`, tal como exige el estándar.

`OAuth2PasswordBearer` no valida el contenido del token: solo lo extrae. La validación la hace una dependencia propia, típicamente `get_current_user`, que decodifica el token y busca al usuario.

## Tokens JWT

Un **JWT** (JSON Web Token) es un token firmado que contiene un payload en JSON. No está cifrado, por lo que no debe incluir datos sensibles, pero su firma impide que sea modificado sin conocer la clave secreta. El payload suele incluir el campo `sub` (el sujeto, normalmente el nombre de usuario) y `exp` (la fecha de expiración). Para generarlos y validarlos se usa una librería como PyJWT: `jwt.encode(payload, SECRET_KEY, algorithm="HS256")` y `jwt.decode(token, SECRET_KEY, algorithms=["HS256"])`.

La clave secreta debe ser larga y aleatoria; puede generarse con `openssl rand -hex 32` y debe leerse desde variables de entorno, nunca escribirse en el código. Si al decodificar el token la firma no coincide o el token expiró, la librería lanza una excepción que la dependencia debe convertir en `HTTPException(status_code=401, detail="Credenciales inválidas", headers={"WWW-Authenticate": "Bearer"})`. Una duración típica para el access token es de 15 a 30 minutos, combinada con un refresh token de mayor duración.

## Hashing de contraseñas

Las contraseñas nunca se guardan en texto plano. Se almacena un **hash** generado con un algoritmo lento y con sal, como **bcrypt** o **Argon2**, mediante librerías como `pwdlib` o `passlib`. Al iniciar sesión se compara el hash de la contraseña recibida con el almacenado usando una función de verificación de tiempo constante. Si el usuario no existe, conviene igualmente ejecutar una verificación ficticia para que el tiempo de respuesta no revele qué usuarios están registrados (protección contra *timing attacks*).

## Scopes

OAuth2 define **scopes**: permisos con nombre como `items:read` o `users:write`. Se declaran en `OAuth2PasswordBearer(tokenUrl="token", scopes={"items:read": "Leer items"})` y se exigen en las operaciones con `Security(get_current_user, scopes=["items:read"])`. Dentro de la dependencia se recibe un objeto `SecurityScopes` con los scopes requeridos por esa operación, que se comparan con los incluidos en el token. Si falta alguno, se responde **403 Forbidden** (autenticado pero sin permiso), a diferencia del 401 (no autenticado).

## Otros esquemas

- `HTTPBasic`: autenticación básica con usuario y contraseña en cada petición; útil para herramientas internas. Para comparar credenciales se recomienda `secrets.compare_digest`.
- `HTTPBearer`: extrae un token Bearer sin asumir el flujo OAuth2.
- `APIKeyHeader`, `APIKeyQuery` y `APIKeyCookie`: leen una API key desde un header (por ejemplo `X-API-Key`), un parámetro de query o una cookie.

## CORS

Cuando un frontend servido desde otro origen (otro dominio, puerto o protocolo) llama a la API, el navegador aplica la política **CORS**. Se habilita agregando `CORSMiddleware` con `app.add_middleware(CORSMiddleware, allow_origins=["https://mi-frontend.com"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])`. Usar `allow_origins=["*"]` junto con `allow_credentials=True` no está permitido por el estándar: si se necesitan cookies o headers de autorización, deben listarse los orígenes explícitamente.

## Checklist de seguridad

1. Servir siempre la API detrás de HTTPS en producción.
2. Mantener `SECRET_KEY` fuera del repositorio y rotarla periódicamente.
3. Usar expiraciones cortas en los access tokens.
4. Distinguir correctamente 401 (falta autenticación) de 403 (falta permiso).
5. Hashear contraseñas con bcrypt o Argon2, nunca con MD5 o SHA-1.
6. Limitar los orígenes CORS a los dominios conocidos.
