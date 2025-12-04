"""
Técnica 2: TEXT-BASED - Conversión a strings numéricos.

Implementación PURA basada en LLMTime (Gruver et al., 2023), técnica ORIGINAL que
convierte series temporales a representación textual procesable por LLMs.

Referencias:
    - Gruver et al. (2023): Large Language Models Are Zero-Shot Time Series Forecasters
    - Repo oficial: https://github.com/ngruver/llmtime
    - Paper: https://arxiv.org/abs/2310.07820
"""

import numpy as np
from typing import Union


def text_based_tokenize(series: np.ndarray,
                        base: int = 10,
                        precision: int = 2,
                        separator: str = ' ') -> dict:
    """
    Tokeniza serie temporal convirtiéndola a representación textual.

    Implementación PURA de LLMTime (Gruver et al., 2023). Convierte cada valor
    numérico a string en base N con precisión decimal especificada, separando
    dígitos para que LLM pueda procesarlos secuencialmente.

    Proceso:
        1. Descomponer valor en signo + dígitos en base N
        2. Separar dígitos con delimiter (ej: 23.45 → " 2 3 . 4 5")
        3. Concatenar valores separados por espacios adicionales

    Args:
        series: Serie temporal univariada [T]
        base: Base numérica (default: 10 decimal)
            - base=10: estándar, compatible con vocabulario LLM
            - base=2: binario, mayor compresión
        precision: Dígitos decimales (default: 2)
        separator: Caracter separador entre dígitos (default: espacio)

    Returns:
        Diccionario con:
            'text': String completo de la serie
            'tokens_per_value': Lista con strings por cada timestep
            'num_tokens': Total de caracteres en texto
            'compression_ratio': T / num_tokens
            'vocabulary_size': base + 3 (dígitos + signo + punto + separador)

    Ejemplo:
        >>> series = np.array([23.45, -12.30, 5.67])
        >>> result = text_based_tokenize(series, base=10, precision=2)
        >>> result['tokens_per_value'][0]
        ' 2 3 . 4 5'
        >>> result['tokens_per_value'][1]
        '- 1 2 . 3 0'
        >>> result['text']
        ' 2 3 . 4 5  - 1 2 . 3 0   5 . 6 7'
    """
    T = len(series)

    # Validaciones
    if base < 2 or base > 36:
        raise ValueError(f"base debe estar en [2, 36], recibido: {base}")
    if precision < 0:
        raise ValueError(f"precision debe ser >= 0, recibido: {precision}")
    if T == 0:
        raise ValueError("Serie vacía")

    # Procesar cada valor de la serie
    tokens_per_value = []
    for value in series:
        token_str = _value_to_text(value, base, precision, separator)
        tokens_per_value.append(token_str)

    # Concatenar todos los valores con doble espacio como separador de timesteps
    text = '  '.join(tokens_per_value)

    # Calcular estadísticas
    num_tokens = len(text)
    vocabulary_size = base + 3  # dígitos + '-' + '.' + separator

    return {
        'text': text,
        'tokens_per_value': tokens_per_value,
        'num_tokens': num_tokens,
        'compression_ratio': T / num_tokens if num_tokens > 0 else 0.0,
        'vocabulary_size': vocabulary_size
    }


def _value_to_text(value: float,
                   base: int,
                   precision: int,
                   separator: str) -> str:
    """
    Convierte valor numérico a representación textual en base N.

    Descompone valor en signo + dígitos antes/después del punto decimal,
    separando cada dígito con el separador especificado.

    Args:
        value: Valor numérico a convertir
        base: Base numérica
        precision: Dígitos decimales
        separator: Separador entre dígitos

    Returns:
        String con representación textual (ej: " 2 3 . 4 5" para 23.45)
    """
    # Manejar signo
    sign_str = '' if value >= 0 else '-'
    abs_value = abs(value)

    # Redondear a precisión especificada
    abs_value = round(abs_value, precision)

    # Separar parte entera y decimal
    integer_part = int(abs_value)
    decimal_part = abs_value - integer_part

    # Convertir parte entera a base N
    if integer_part == 0:
        integer_digits = [0]
    else:
        integer_digits = []
        temp = integer_part
        while temp > 0:
            integer_digits.append(temp % base)
            temp //= base
        integer_digits.reverse()

    # Convertir dígitos a strings
    integer_str = separator.join([_digit_to_char(d, base) for d in integer_digits])

    # Construir string completo
    if precision > 0:
        # Convertir parte decimal a base N
        decimal_digits = []
        temp = decimal_part
        for _ in range(precision):
            temp *= base
            digit = int(temp)
            decimal_digits.append(digit)
            temp -= digit

        decimal_str = separator.join([_digit_to_char(d, base) for d in decimal_digits])
        result = f"{sign_str}{separator}{integer_str}{separator}.{separator}{decimal_str}"
    else:
        result = f"{sign_str}{separator}{integer_str}"

    return result


def _digit_to_char(digit: int, base: int) -> str:
    """
    Convierte dígito numérico a caracter.

    Args:
        digit: Dígito en rango [0, base-1]
        base: Base numérica

    Returns:
        Caracter: '0'-'9' para dígitos 0-9, 'a'-'z' para dígitos 10-35
    """
    if digit < 10:
        return str(digit)
    else:
        return chr(ord('a') + digit - 10)


def decode_text_based(text: str,
                      base: int = 10,
                      separator: str = ' ') -> np.ndarray:
    """
    Decodifica texto a valores numéricos (reconstrucción).

    Proceso inverso de text_based_tokenize: parsea strings de dígitos
    separados y reconstruye valores numéricos originales.

    Args:
        text: String completo de la serie
        base: Base numérica usada en encoding
        separator: Separador usado entre dígitos

    Returns:
        Serie reconstruida [T]
    """
    # Separar por doble espacio (separador de timesteps)
    value_strings = text.split('  ')

    # Decodificar cada valor
    values = []
    for val_str in value_strings:
        if not val_str.strip():
            continue

        # Parsear signo
        is_negative = val_str.strip().startswith('-')
        val_str_clean = val_str.replace('-', '').strip()

        # Separar parte entera y decimal
        parts = val_str_clean.split('.')

        # Decodificar parte entera
        integer_digits_str = parts[0].strip().split(separator)
        integer_digits = [_char_to_digit(d.strip()) for d in integer_digits_str if d.strip()]
        integer_value = sum(d * (base ** (len(integer_digits) - 1 - i))
                          for i, d in enumerate(integer_digits))

        # Decodificar parte decimal si existe
        decimal_value = 0.0
        if len(parts) > 1:
            decimal_digits_str = parts[1].strip().split(separator)
            decimal_digits = [_char_to_digit(d.strip()) for d in decimal_digits_str if d.strip()]
            decimal_value = sum(d * (base ** (-(i + 1)))
                               for i, d in enumerate(decimal_digits))

        # Combinar
        value = integer_value + decimal_value
        if is_negative:
            value = -value

        values.append(value)

    return np.array(values)


def _char_to_digit(char: str) -> int:
    """
    Convierte caracter a dígito numérico.

    Args:
        char: Caracter '0'-'9' o 'a'-'z'

    Returns:
        Dígito en rango [0, 35]
    """
    if char.isdigit():
        return int(char)
    else:
        return ord(char.lower()) - ord('a') + 10


def visualize_text_based(series: np.ndarray,
                         tokens_per_value: list) -> dict:
    """
    Genera información para visualizar tokenización text-based.

    Args:
        series: Serie original [T]
        tokens_per_value: Lista de strings por timestep

    Returns:
        Diccionario con información para plotting
    """
    # Reconstruir serie desde texto
    text = '  '.join(tokens_per_value)
    reconstructed = decode_text_based(text)

    # Calcular longitudes de tokens
    token_lengths = [len(tok) for tok in tokens_per_value]

    return {
        'series': series,
        'reconstructed': reconstructed,
        'tokens_per_value': tokens_per_value,
        'token_lengths': token_lengths,
        'avg_token_length': np.mean(token_lengths),
        'total_characters': sum(token_lengths)
    }
