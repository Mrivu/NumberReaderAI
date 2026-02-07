# NumberReaderAI
TKT20010 Aineopintojen harjoitustyö: Algoritmit ja tekoäly

# Interface guide
Run the interface with:
```
poetry run python3 interface.py 
```
The interface allows the user to create and pass data through networks.

### Simple guide
In the interface many:
- Run the getw command to load trained weights and biases.
- Run the generate command to setup a network.
- Run the load command to get a random image.
- Run the ff command to see if the network guesses it.

#### generate
This command creates a new network with the following layers: [784, 16, 16, 10].
It also gives each weight and bias a random value.
#### ff
Feed loaded image through neural network and see the output.
#### train
Train the network. Higher percentages may require multiple trains.
#### test
Test the network's accuracy.
#### savew
Save the current weights and biases.
#### getw
Load existing weights and biases.
#### load
Load stores a random image from the test database. This image can be viewed and removed from storing. If an image is loaded, the image will be used as data in feeding forward.
#### view
Display what the image looks like and a label for what number it is supposed to be.
#### remove
Unload a loaded image. Removing when an image is not loaded does nothing.

## Testing

### Run tests
poetry run coverage run --branch -m pytest tests

### Get coverage
poetry run coverage report -m

### Run Pylint
poetry run pylint .

### Coverage
![coverage](coverage.png)
