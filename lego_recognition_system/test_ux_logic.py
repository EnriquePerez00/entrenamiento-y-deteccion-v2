import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.part_resolver import resolve_set, resolve_piece
from src.logic.model_registry import get_training_status, filter_pending

def simulate_ux(mode="set", value="75078-1", num_parts=5):
    print(f"\n--- SIMulando UX INTERACTIVO (Modo: {mode}, Valor: {value}) ---")
    
    # 1. Resolver partes
    try:
        if mode == "set":
            resolved = resolve_set(value, max_parts=num_parts)
        else:
            resolved = resolve_piece(value)
    except Exception as e:
        print(f"❌ Error al resolver: {e}")
        return

    print(f"✅ Piezas resueltas ({len(resolved)}):")
    for p in resolved:
        print(f"   • {p['ldraw_id']} - {p['name']}")

    # 2. Verificar estado de modelos
    # Simulamos que no hay Drive service localmente, solo chequeo local
    status = get_training_status(resolved, PROJECT_ROOT)
    
    print("\n" + "─" * 60)
    print(f"{'Pieza':<15} {'Nombre':<30} {'Estado'}")
    print("─" * 60)
    for s in status:
        icon = "✅ Ya entrenado" if s['is_complete'] else "❌ Pendiente"
        print(f"{s['ldraw_id']:<15} {s['name'][:28]:<30} {icon}")
    print("─" * 60)

    pending = filter_pending(status)
    if not pending:
        print("\n🎉 ¡Todo listo! No hay piezas pendientes.")
    else:
        print(f"\n🚀 Piezas que se enviarían a entrenar: {[p['ldraw_id'] for p in pending]}")

if __name__ == "__main__":
    # Test 1: Set con 3 piezas random
    simulate_ux(mode="set", value="75078-1", num_parts=3)
    
    # Test 2: Pieza específica
    simulate_ux(mode="piece", value="14769")
