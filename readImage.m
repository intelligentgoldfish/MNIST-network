%%readImage.m
%%Reformats and displays 784x1 column vector and displays as image

function readImage(imageVector)
    image = transpose(reshape(imageVector,28,28));
    imagesc(image);
    %colormap(gray);
end