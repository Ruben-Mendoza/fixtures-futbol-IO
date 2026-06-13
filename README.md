# fixtures-futbol-IO
Generación de fixtures de fútbol con BILP y Branch &amp; Bound · Investigación Operativa · FAMAF-UNC

# Generación de Fixtures de Fútbol mediante Programación Lineal Entera Binaria

> Proyecto final — **Investigación Operativa**  
> Licenciatura en Matemática Aplicada · FAMAF, Universidad Nacional de Córdoba
> 
---

## Descripción

Este proyecto aborda el problema de generación de calendarios (fixtures) para torneos de fútbol del tipo todos-contra-todos de una ronda. Se formula un modelo de Programación Lineal Entera Binaria (BILP) con función objetivo que minimiza simultáneamente dos criterios:

- **Equidad deportiva:** minimización de *breaks* de localía (secuencias consecutivas como local o visitante).
- **Eficiencia logística:** minimización de la distancia total recorrida por equipos que juegan de visitante en fechas consecutivas.

Se implementa en Python el algoritmo exacto de **Branch and Bound (B&B)** y se compara contra la heurística **Round Robin** en términos de calidad de solución y tiempo de cómputo, para instancias de N ∈ {4, 6, 8} equipos argentinos con distancias reales entre estadios (calculadas con la fórmula de Haversine).

---

## Tecnologías y herramientas

- **Lenguaje:** Python 3
- **Bibliotecas:** `numpy`, `itertools`, `multiprocessing`
- **Modelado:** formulación BILP propia (sin solver externo)
- **Paralelización:** `multiprocessing` para instancias N=8
- **Documentación:** LaTeX

---
