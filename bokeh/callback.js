var data = source.data;
var f = slider.value;
var x = data['x']
var y = data['y']
for (var i = 0; i < x.length; i++) {
    y[i] = Math.pow(x[i], f)
}

// necessary becasue we mutated source.data in-place
source.change.emit();