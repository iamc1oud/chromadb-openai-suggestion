import chromadb
import json
import numpy as np
from typing import List, Dict, Any

class MenuRecommendationSystem:
    def __init__(self):
        # Initialize ChromaDB client
        self.client = chromadb.Client()
        # Create collection for menu items
        self.collection = self.client.create_collection(name="menu_items")
        
    def load_menu_data(self, menu_data: Dict[str, Any]):
        """Load menu data into ChromaDB collection."""
        items = []
        ids = []
        metadatas = []
        
        # Extract all menu items from categories
        for category in menu_data["data"]["menuCategories"]:
            category_name = category["menuCategoryName"]
            category_id = category["menuCategoryId"]
            
            for item in category["items"]:
                item_id = str(item["id"])
                item_name = item["name"]
                item_price = item["price"]
                
                # TODO: Create embeddings (in a real system, use a text embedding model)
                # Here we'll use the item name as the document for simplicity
                items.append(item_name)
                ids.append(item_id)
                metadatas.append({
                    "name": item_name,
                    "price": item_price,
                    "category_name": category_name,
                    "category_id": category_id,
                    "hotel_id": item["hotel"]
                })
        
        # Add items to collection
        if items:
            self.collection.add(
                documents=items,
                ids=ids,
                metadatas=metadatas
            )
        
        print(f"Loaded {len(items)} menu items into ChromaDB")
    
    def get_cart_item_names(self, cart_data: Dict[str, Any]) -> List[str]:
        """Extract item names from cart data."""
        item_names = []
        
        for cart_item in cart_data["data"]["validItems"]:
            item_names.append(cart_item["item"]["name"])
            
        return item_names
    
    def recommend_items(self, cart_data: Dict[str, Any], num_recommendations: int = 3) -> List[Dict[str, Any]]:
        """Recommend items based on cart contents."""
        cart_item_names = self.get_cart_item_names(cart_data)
        
        # Skip if cart is empty
        if not cart_item_names:
            return []
        
        # Use the first cart item as query
        # In a real system, you might combine all cart items or use a more sophisticated approach
        # This is a simplified example
        query_results = []
        
        # Query for each item in the cart and combine results
        for cart_item in cart_item_names:
            results = self.collection.query(
                query_texts=[cart_item],
                n_results=num_recommendations + len(cart_item_names)  # Get extra to account for items already in cart
            )
            
            # Add to our combined results
            for i, item_id in enumerate(results["ids"][0]):
                query_results.append({
                    "id": int(item_id),
                    "name": results["metadatas"][0][i]["name"],
                    "price": results["metadatas"][0][i]["price"],
                    "category": results["metadatas"][0][i]["category_name"],
                    "similarity": results["distances"][0][i] if "distances" in results else 0
                })
        
        # Deduplicate results
        seen_ids = set()
        unique_results = []
        
        for item in query_results:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                unique_results.append(item)
        
        # Filter out items that are already in the cart
        cart_item_ids = [item["item"]["id"] for item in cart_data["data"]["validItems"]]
        recommendations = [item for item in unique_results if item["id"] not in cart_item_ids]
        
        # Sort by similarity score (lower is better in ChromaDB's case)
        recommendations.sort(key=lambda x: x["similarity"])
        
        # Return top N recommendations
        return recommendations[:num_recommendations]

    def recommend_by_cross_sell(self, cart_data: Dict[str, Any], menu_data: Dict[str, Any], num_recommendations: int = 3) -> List[Dict[str, Any]]:
        """Recommend items based on cross-sell relationships in the cart data."""
        # Get cross-sell item IDs from cart items
        cross_sell_ids = set()
        cart_item_ids = set()
        
        for cart_item in cart_data["data"]["validItems"]:
            cart_item_ids.add(cart_item["item"]["id"])
            
            # Add cross-sell items if they exist
            if "crossSellItems" in cart_item["item"] and cart_item["item"]["crossSellItems"]:
                for cross_sell_id in cart_item["item"]["crossSellItems"]:
                    cross_sell_ids.add(cross_sell_id)
        
        # Remove items already in cart
        cross_sell_ids = cross_sell_ids - cart_item_ids
        
        # Find the matching items in menu data
        recommendations = []
        for category in menu_data["data"]["menuCategories"]:
            for item in category["items"]:
                if item["id"] in cross_sell_ids:
                    recommendations.append({
                        "id": item["id"],
                        "name": item["name"],
                        "price": item["price"],
                        "category": category["menuCategoryName"],
                        "recommendation_type": "cross_sell"
                    })
                    
                    # Break early if we have enough recommendations
                    if len(recommendations) >= num_recommendations:
                        break
            
            if len(recommendations) >= num_recommendations:
                break
        
        return recommendations[:num_recommendations]
    
    def get_todays_offers(self, menu_data: Dict[str, Any], max_items: int = 3) -> List[Dict[str, Any]]:
        """Get today's special offers from the menu."""
        offers = []
        
        # Find "Today's Offers" category
        for category in menu_data["data"]["menuCategories"]:
            if category["menuCategoryName"] == "Today's Offers":
                # Get items from this category
                for item in category["items"]:
                    offers.append({
                        "id": item["id"],
                        "name": item["name"],
                        "price": item["price"],
                        "category": "Today's Offers",
                        "recommendation_type": "daily_special"
                    })
                
                # Limit to max_items
                offers = offers[:max_items]
                break
        
        return offers
    
    def get_recommendations(self, cart_data: Dict[str, Any], menu_data: Dict[str, Any], num_recommendations: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get comprehensive recommendations combining different strategies."""
        # First try direct cross-sell recommendations from cart data
        cross_sell_recs = self.recommend_by_cross_sell(cart_data, menu_data, num_recommendations=2)
        
        # Get similarity-based recommendations 
        similarity_recs = self.recommend_items(cart_data, num_recommendations=2)
        
        # Get today's offers
        todays_offers = self.get_todays_offers(menu_data, max_items=1)
        
        # Combine all recommendations
        all_recommendations = {
            "cross_sell_recommendations": cross_sell_recs,
            "similar_items": similarity_recs,
            "daily_specials": todays_offers
        }
        
        return all_recommendations

# Example usage
def main():
    # Load cart and menu data
    with open('cart.json', 'r') as f:
        cart_data = json.load(f)
    
    with open('short_menu.json', 'r') as f:
        menu_data = json.load(f)
    
    # Initialize recommendation system
    recommender = MenuRecommendationSystem()
    
    # Load menu data
    recommender.load_menu_data(menu_data)
    
    # Get recommendations
    recommendations = recommender.get_recommendations(cart_data, menu_data)
    
    # Print recommendations
    print("Recommendations for your cart:")
    print("\nBased on cross-sell relationships:")
    for item in recommendations["cross_sell_recommendations"]:
        print(f"- {item['name']} (₹{item['price']})")
    
    print("\nSimilar items you might like:")
    for item in recommendations["similar_items"]:
        print(f"- {item['name']} (₹{item['price']})")
    
    print("\nToday's special offers:")
    for item in recommendations["daily_specials"]:
        print(f"- {item['name']} (₹{item['price']})")

if __name__ == "__main__":
    main()