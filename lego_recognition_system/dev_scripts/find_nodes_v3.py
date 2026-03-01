import bpy
import nodeitems_utils

def list_categories():
    print("LISTING COMPOSITING CATEGORIES")
    try:
        categories = list(nodeitems_utils.node_categories_iter('COMPOSITING'))
        for cat in categories:
            print(f"CATEGORY: {cat.label} ({cat.identifier})")
            for item in cat.items(bpy.context):
                 if hasattr(item, "nodetype"):
                     print(f"  NODE: {item.nodetype} ({item.label})")
                 else:
                     print(f"  ITEM: {item}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    list_categories()
