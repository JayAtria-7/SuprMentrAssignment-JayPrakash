'''
shopping cart:
create shopping cart using list:
add items,remove one item, print total items

'''

cart=[]

cart.append("Cabbage")
cart.append("Flowers")
cart.append("Chips")
cart.append("Drinks")
print("Cart before removing")
print(cart)

cart.pop()
print("Cart after removing")
print(cart)

print("Total items",len(cart))