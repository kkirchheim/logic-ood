% knowledge base for primatenet

monkey(Primate) :- old_world_monkey(Primate).
monkey(Primate) :- new_world_monkey(Primate).

ape(Primate) :- lesser_ape(Primate).
ape(Primate) :- great_ape(Primate).

primate(Primate) :- monkey(Primate).
primate(Primate) :- ape(Primate).
primate(Primate) :- lemur(Primate).

old_world_monkey(guenon).
old_world_monkey(colobus).
old_world_monkey(baboon).
old_world_monkey(langur).
old_world_monkey(macaque).

new_world_monkey(squirrel_monkey).
new_world_monkey(spider_monkey).
new_world_monkey(howler_monkey).
new_world_monkey(marmoset).
new_world_monkey(titi).
new_world_monkey(capuchin).

lesser_ape(gibbon).
lesser_ape(siamang).

great_ape(chimpanzee).
great_ape(orangutan).

lemur(madagascar_cat).

is_primate(Primate, true) :- primate(Primate).

% some helpers
consistent_with_kb(Entity, Category, true) :-
    call(Category, Entity).

consistent_with_kb(Entity, Category, false) :-
    \+ call(Category, Entity).
