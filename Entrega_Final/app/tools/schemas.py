"""Contratos Pydantic de las herramientas de los agentes.

- Las clases `*Input` son el `args_schema` de cada herramienta: LangChain valida los
  argumentos que propone el LLM antes de ejecutar nada. Si no cumplen, el agente recibe el
  error como mensaje y puede corregir la llamada (el grafo no se rompe).
- Las clases de salida son el *artifact* de cada herramienta (`response_format=
  "content_and_artifact"`): el LLM lee un texto breve y el sistema conserva el dato
  estructurado en el `ToolMessage`, del que se extraen, por ejemplo, las fuentes reales.
"""

from pydantic import BaseModel, Field, field_validator

# --- Investigador ---


class BuscarDocumentosInput(BaseModel):
    consulta: str = Field(min_length=3, max_length=300, description="Qué buscar, en lenguaje natural o palabras clave")
    k: int = Field(default=4, ge=1, le=8, description="Cantidad de fragmentos a devolver (1 a 8)")

    @field_validator("consulta")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("la consulta debe tener al menos 3 caracteres útiles")
        return value


class Fragmento(BaseModel):
    fuente: str = Field(description="Archivo del corpus")
    seccion: str = Field(default="", description="Título de la sección Markdown")
    chunk_id: str
    puntaje_rrf: float = Field(ge=0)
    contenido: str


class ResultadoBusqueda(BaseModel):
    consulta: str
    fragmentos: list[Fragmento]

    @property
    def fuentes(self) -> list[str]:
        return list(dict.fromkeys(f.fuente for f in self.fragmentos))

    def to_text(self) -> str:
        if not self.fragmentos:
            return "Sin resultados para esa búsqueda."
        return "\n\n---\n\n".join(f"[Fuente: {f.fuente}]\n{f.contenido}" for f in self.fragmentos)


class ListaDocumentos(BaseModel):
    documentos: list[str]


# --- Analista ---


class CalcularInput(BaseModel):
    expresion: str = Field(
        min_length=1, max_length=300, description="Expresión aritmética con punto decimal, p. ej. '(28 - 14) / 14 * 100'"
    )


class ResultadoCalculo(BaseModel):
    expresion: str
    resultado: float


class EstadisticasInput(BaseModel):
    valores: list[float] = Field(min_length=1, max_length=1000, description="Lista de números")


class Estadisticas(BaseModel):
    n: int = Field(ge=1)
    suma: float
    promedio: float
    mediana: float
    minimo: float
    maximo: float
    desvio_estandar: float = Field(ge=0)


class VariacionInput(BaseModel):
    valor_inicial: float = Field(description="Valor de referencia (no puede ser 0)")
    valor_final: float

    @field_validator("valor_inicial")
    @classmethod
    def _not_zero(cls, value: float) -> float:
        if value == 0:
            raise ValueError("el valor inicial es 0: la variación porcentual no está definida")
        return value


class ResultadoVariacion(BaseModel):
    valor_inicial: float
    valor_final: float
    variacion_porcentual: float


class SentimientoInput(BaseModel):
    textos: list[str] = Field(min_length=1, max_length=200, description="Textos completos a analizar")

    @field_validator("textos")
    @classmethod
    def _no_blank(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values if v.strip()]
        if not cleaned:
            raise ValueError("todos los textos están vacíos")
        return cleaned


class SentimientoTexto(BaseModel):
    texto: str
    puntaje: float = Field(ge=-1, le=1)
    etiqueta: str
    positivas: list[str]
    negativas: list[str]


class ResultadoSentimiento(BaseModel):
    n: int
    puntaje_promedio: float = Field(ge=-1, le=1)
    distribucion: dict[str, int]
    porcentaje_positivo: float = Field(ge=0, le=100)
    detalle: list[SentimientoTexto]


# --- Publicación (efecto lateral, solo tras aprobación humana) ---


class PublicarInformeInput(BaseModel):
    job_id: str = Field(min_length=1, max_length=100)
    consulta: str = Field(min_length=1)
    contenido: str = Field(min_length=1, description="Respuesta final validada")
    aprobado_por: str = Field(default="", max_length=500)


class InformePublicado(BaseModel):
    ruta: str
    publicado_en: str
    bytes: int = Field(ge=0)


class ErrorHerramienta(BaseModel):
    error: str
