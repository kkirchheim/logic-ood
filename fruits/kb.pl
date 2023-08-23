% knowledge base for fruits

fruit(orange).
fruit(strawberry).
fruit(blueberry).
fruit(pepper_green).
fruit(grape_blue).
fruit(lemon).
fruit(apple_granny_smith).
fruit(papaya).
fruit(tomato).
fruit(apple_braeburn).
fruit(cactus_fruit).
fruit(peach).
fruit(apricot).
fruit(watermelon).
fruit(pineapple).
fruit(banana).
fruit(cantaloupe).
fruit(cucumber_ripe).
fruit(cherry).
fruit(corn).
fruit(clementine).
fruit(mango).
fruit(plum).
fruit(limes).
fruit(potato_red).
fruit(avocado).
fruit(pear).
fruit(passion_fruit).
fruit(pomegranate).
fruit(onion_white).
fruit(pepper_red).
fruit(kiwi).
fruit(raspberry).


color(apricot, orange).
color(peach, orange).
color(avocado, green).
color(corn, yellow).
color(clementine, orange).
color(pepper_red, red).
color(cantaloupe, yellow).
color(tomato, red).
color(cucumber_ripe, brown).
color(kiwi, brown).
color(pineapple, brown).
color(blueberry, black).
color(papaya, green).
color(apple_braeburn, red).
color(plum, red).
color(onion_white, brown).
color(banana, yellow).
color(passion_fruit, black).
color(limes, green).
color(raspberry, red).
color(pomegranate, red).
color(potato_red, brown).
color(mango, green).
color(grape_blue, black).
color(lemon, yellow).
color(strawberry, red).
color(apple_granny_smith, green).
color(pepper_green, green).
color(cactus_fruit, green).
color(pear, green).
color(cherry, red).
color(watermelon, red).
color(orange, orange).


is_fruit(Fruit, true) :- fruit(Fruit).

is_sat(Fruit, Color, IsFruit) :-
    fruit(Fruit),
    color(Fruit, Color),
    is_fruit(Fruit, IsFruit).
