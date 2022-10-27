gaussian_kernel = function(x_1, x_2, gamma) {
    n_row = nrow(x_1)
    n_col = nrow(x_2)
    l_matrix = matrix(seq(1, by = 0, length = ncol(x_1)), ncol = 1)
    m = matrix(seq(1, by = 0, length = n_row), nrow = 1)
    n = matrix(seq(1, by = 0, length = n_col), nrow = 1)
    val1 = (x_1 * x_1) %*% l_matrix %*% n
    val2 = t((x_2 * x_2) %*% l_matrix %*% m)
    val3 = x_1 %*% t(x_2)
    result = exp(-(1 / gamma) * (val1 + val2 - 2 * val3))
    return(result)
}

I_function = function(n_row) {
    y = matrix(seq(0, by = 0, length = n_row^2), nrow = n_row, ncol = n_row)
    for (i in 1:n_row) {
        y[i, i] = 1
    }
    return(y)
}

obtained_weight_matrix = function(data_labels, original_labels_matrix, num_classes) {
    present_labels_data_number = t(as.matrix(apply(data_labels, 2, sum)))
    original_labels_matrix = original_labels_matrix + present_labels_data_number
    the_matrix_init = numeric(length = num_classes)
    for (i in 1:num_classes) {
        the_matrix_init[i] = 1 / original_labels_matrix[i]
    }
    the_matrix_init = matrix(the_matrix_init, ncol = 1)
    the_matrix_next = data_labels %*% the_matrix_init
    num_samples = nrow(the_matrix_next)
    the_matrix_final = matrix(seq(0, by = 0, length = (num_samples^2)), nrow = num_samples, ncol = num_samples)
    for (i in 1:num_samples) {
        the_matrix_final[i, i] = the_matrix_next[i, 1]
    }
    return(list(original_labels_matrix, the_matrix_final))
}

normial = function(x) {
    return((2 * (x - min(x)) / (max(x) - min(x))) - 1)
}

obtained_acc_G_mean = function(x) {
    the_sum = 0
    the_G_mean = 1
    for (i in 1:nrow(x)) {
        the_sum = the_sum + x[i, i]
        the_G_mean = the_G_mean * (x[i, i] / sum(x[i,]))
    }
    the_acc = the_sum / sum(x)
    the_G_mean = the_G_mean^(1 / nrow(x))
    return(list(the_acc * 100, the_G_mean * 100))
}

model = function(c, gamma, num_classes, max_sample, train_path, samples_number = 0, test_path = "src", single = FALSE, init_num = 100) {
    if (single) {
        total_data = read.table(train_path, header = TRUE, sep = ",", stringsAsFactors = TRUE)
    } else {
        data_train = read.table(train_path, header = TRUE, sep = ",", stringsAsFactors = TRUE)
        data_test = read.table(test_path, header = TRUE, sep = ",", stringsAsFactors = TRUE)
        total_data = rbind(data_train, data_test)
        samples_number = nrow(data_train)
    }
    variables_number = ncol(total_data) - 1
    total_data$label = as.numeric(total_data$label)
    total_data_normial = as.data.frame(lapply(total_data[, c(1:variables_number)], normial))
    total_data = cbind(total_data_normial, total_data[variables_number + 1])
    data = total_data[c(1:samples_number),]
    testing_data = total_data[-c(1:samples_number),]
    original_labels_matrix = matrix(seq(0, by = 0, length = num_classes), nrow = 1, ncol = num_classes)
    init_indic = sample(1:samples_number, sample(1:init_num, 1), replace = FALSE)
    training_data = data[init_indic,]
    while (length(unique(training_data[, variables_number + 1])) != num_classes) {
        init_indic = sample(1:samples_number, sample(1:init_num, 1), replace = FALSE)
        training_data = data[init_indic,]
    }
    data = data[-init_indic,]
    training_data_variables = as.matrix(training_data[, c(1:variables_number)])
    retain_data = training_data_variables
    retain_samples = nrow(retain_data)
    instances_labels = training_data[, variables_number + 1]
    categories = unique(instances_labels)
    training_data_labels = as.data.frame(matrix(seq(0, by = 0, length = nrow(training_data) * num_classes), nrow = nrow(training_data), ncol = num_classes))
    names(training_data_labels) = categories
    for (i in 1:num_classes) {
        position = which(instances_labels == categories[i])
        training_data_labels[position, i] = 1
    }
    training_data_labels = as.matrix(training_data_labels)
    weight_result = obtained_weight_matrix(training_data_labels, original_labels_matrix, num_classes)
    original_labels_matrix = weight_result[[1]]
    K_0 = gaussian_kernel(training_data_variables, training_data_variables, gamma)
    R = solve(K_0 + solve(weight_result[[2]]) / c)
    Alpha = R %*% training_data_labels

    start_num = 1
    end_num = sample(1:init_num, 1)
    while (start_num <= nrow(data)) {
        training_data = data[c(start_num:end_num),]
        training_data_variables = as.matrix(training_data[, c(1:variables_number)])
        instances_labels = training_data[, variables_number + 1]
        training_data_labels = as.data.frame(matrix(seq(0, by = 0, length = nrow(training_data) * num_classes), nrow = nrow(training_data), ncol = num_classes))
        names(training_data_labels) = categories
        for (i in 1:num_classes) {
            position = which(instances_labels == categories[i])
            training_data_labels[position, i] = 1
        }
        training_data_labels = as.matrix(training_data_labels)
        weight_result = obtained_weight_matrix(training_data_labels, original_labels_matrix, num_classes)
        original_labels_matrix = weight_result[[1]]
        retain_samples = retain_samples + (end_num - start_num + 1)
        if (retain_samples > max_sample) {
            NP = retain_samples - max_sample
            X_D = retain_data[c(1:NP), ]
            retain_data = retain_data[-c(1:NP),]
            R = R[-c(1:NP), - c(1:NP)] - R[-c(1:NP), c(1:NP)] %*% solve(R[c(1:NP), c(1:NP)]) %*% R[c(1:NP), - c(1:NP)]
            Q = -1 * R %*% gaussian_kernel(retain_data, X_D)
            Alpha = Alpha[-c(1:NP),] + Q %*% Alpha[c(1:NP),]
            retain_samples = max_sample
        }
        K_1 = gaussian_kernel(retain_data, training_data_variables, gamma)
        K_2 = gaussian_kernel(training_data_variables, training_data_variables, gamma)
        Q = -1 * R %*% K_1
        MIU = K_2 + solve(weight_result[[2]]) / c + t(K_1) %*% Q
        Alpha = rbind(Alpha + Q %*% solve(MIU) %*% (training_data_labels - t(K_1) %*% Alpha),
                      solve(MIU) %*% (training_data_labels - t(K_1) %*% Alpha))

        N1 = nrow(retain_data)
        N2 = nrow(training_data)
        R = rbind(cbind(R, matrix(seq(0, by = 0, length = N1 * N2), nrow = N1)), matrix(seq(0, by = 0, length = N2 * (N1 + N2)),
        nrow = N2)) + rbind(Q, I_function(N2)) %*% solve(MIU) %*% t(rbind(Q, I_function(N2)))
        retain_data = rbind(retain_data, training_data_variables)
        start_num = end_num + 1
        end_num = end_num + sample(1:init_num, 1)
        if (end_num > nrow(data)) {
            end_num = nrow(data)
        }
    }

    testing_data_variables = as.matrix(testing_data[, c(1:variables_number)])
    H = gaussian_kernel(testing_data_variables, retain_data, gamma)
    aim = as.data.frame(H %*% Alpha)
    aim_result = aim[, order(as.numeric(colnames(aim)))]
    aim_result$result = 0
    for (i in 1:nrow(aim_result)) {
        aim_result[i, ncol(aim_result)] = which.max(aim_result[i, c(1:(ncol(aim_result) - 1))])
    }
    table0 = table(testing_data$label, aim_result$result)
    final_result <- obtained_acc_G_mean(table0)
    Acc <- final_result[[1]]
    Gmean <- final_result[[2]]
    comp = data.frame(Acc, Gmean)
    names(comp) = c("Acc", "Gmean")
    print(comp)
    saver = read.table("D:/program/data.csv", header = TRUE, sep = ",")
    saver = rbind(saver, comp)
    write.csv(saver, "D:/program/data.csv", row.names = FALSE)
}

for (number in 1:50) {
    model(2^8, 2^-4, 5, 5000, "D:/program/pageblocks_train.csv", 3000, "D:/program/pageblocks_test.csv")
}
