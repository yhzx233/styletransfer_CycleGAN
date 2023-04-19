import data
import train
import model

X_iter, Y_iter = data.load_data_scenery(1)
gen1, gen2 = model.Generator(), model.Generator()
disc1, disc2 = model.Discriminator(), model.Discriminator()

train.train(gen1, gen2, disc1, disc2, X_iter, Y_iter)
