from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, Dropout


def CNN(input_dim,
        input_length,
        vec_size,
        output_shape,
        output_type='multiple'):

    data_input = Input(shape=[input_length])
    word_vec = Embedding(input_dim=input_dim + 1,
                         input_length=input_length,
                         output_dim=vec_size)(data_input)
    x = Conv1D(filters=128,
               kernel_size=[3],
               strides=1,
               padding='same',
               activation='relu')(word_vec)
    x = GlobalMaxPool1D()(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.1)(x)
    if output_type == 'multiple':
        x = Dense(output_shape, activation='sigmoid')(x)
        model = Model(inputs=data_input, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
    elif output_type == 'single':
        x = Dense(output_shape, activation='softmax')(x)
        model = Model(inputs=data_input, outputs=x)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
    else:
        raise ValueError('output_type should be multiple or single')
    return model


if __name__ == '__main__':
    model = CNN(input_dim=10, input_length=10, vec_size=10, output_shape=10, output_type='multiple')
    model.summary()
