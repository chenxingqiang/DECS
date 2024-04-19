function [chromosome] = Calculate_Q(chromosome, whole_size, num_node, num_edge, adjacent_array)
    % calculate the modularity from the genome matrix and chromosome vector, respectively
    % input:
    %       choromosome - all chromosomes in the population
    %       whole_size - the total number of the original and new individuals
    %       num_node - the number of nodes in the network
    %       num_edge - the number of edges in the network
    %       adjacent_array - the adjacent matrix of the network
    % output:
    %       choromosome - all chromosomes in the population

    % transform the genoem matrix into the vector whose elements
    % represent the community to which a node belongs
    [node_chrom] = change(chromosome, whole_size, num_node);

    for pop_id = 1:whole_size
        num_cluster = max(node_chrom(pop_id, :));
        e = zeros(1, num_cluster);
        a = zeros(1, num_cluster);

        for j = 1:num_cluster
            cluster_id = j;
            nodes_in_cluster = find(node_chrom(pop_id, :) == cluster_id); % find the nodes within the same community
            L = length(nodes_in_cluster); % L - the number of nodes in a community

            for k = 1:L

                for m = 1:num_node

                    if adjacent_array(nodes_in_cluster(k), m) == 1 % find the node's neighbors
                        % check if nodes are clustered into the same community
                        if chromosome(pop_id).genome(nodes_in_cluster(k), m) == 1
                            e(cluster_id) = e(cluster_id) + 1;
                        else
                            a(cluster_id) = a(cluster_id) + 1;
                        end

                    end

                end

            end

        end

        e = e ./ 2;
        a = a ./ 2;
        a = a + e;
        e = e / num_edge;
        a = (a / num_edge) .^ 2;
        Q = 0;

        for n = 1:num_cluster
            Q = Q + e(n) - a(n);
        end

        chromosome(pop_id).fitness_1 = Q; % modularity calculated from the genome matrix
        chromosome(pop_id).clusters = node_chrom(pop_id, :); % the clustering result
        chromosome(pop_id).fitness_2 = Modularity(adjacent_array, chromosome(pop_id).clusters); % modularity
    end

end

function [edge_begin_end] = CreatEdgeList(adjacent_array)
    % this function is used to create the edge list

    [x, y, ~] = find (adjacent_array);
    l = length(x);
    edge_begin_end = zeros(l / 2, 3);
    j = 0;

    for i = 1:l

        if x(i) > y(i)
            j = j + 1;
            edge_begin_end(j, 1) = j;
            edge_begin_end(j, 2) = x(i);
            edge_begin_end(j, 3) = y(i);
        end

    end

end

function [child_chromosome_1, child_chromosome_2] = Crossover(chromosome, selected_pop_id, node_num)
    % a pair of selected individuals to generate 2 children individuals
    child_chromosome_1.genome = [];
    child_chromosome_1.clusters = [];
    child_chromosome_1.fitness_1 = [];
    child_chromosome_1.fitness_2 = [];
    child_chromosome_2.genome = [];
    child_chromosome_2.clusters = [];
    child_chromosome_2.fitness_1 = [];
    child_chromosome_2.fitness_2 = [];
    cross_over_population_id = selected_pop_id;
    cross_over_count = randi(round(node_num / 2), 1);
    cross_over_node_id = randi(node_num, cross_over_count, 1);

    % two-way crossover
    temp1_part = chromosome(cross_over_population_id(1)).genome(cross_over_node_id, :);
    temp2_part = chromosome(cross_over_population_id(2)).genome(cross_over_node_id, :);
    temp1_whole = chromosome(cross_over_population_id(1)).genome;
    temp2_whole = chromosome(cross_over_population_id(2)).genome;
    temp1_whole(cross_over_node_id, :) = temp2_part;
    temp1_whole(:, cross_over_node_id) = temp2_part';
    temp2_whole(cross_over_node_id, :) = temp1_part;
    temp2_whole(:, cross_over_node_id) = temp1_part';

    child_chromosome_1.genome = temp1_whole; % child 1
    child_chromosome_2.genome = temp2_whole; % child 2

end

function [Mod, chromosome, Division, time] = DECS_1(adjacent_array, ...
        maxgen, pop_size, p_mutation, p_migration, p_mu_mi, PGLP_iter)
    % detect the community structure at the 1st time step

    % input:  adjacent_array - the adjacent matrix
    %         maxgen - the maximum number of iterations
    %         pop_size - the population size
    %         p_mutation - the mutation rate
    %         p_migration - the migration rate
    %         p_mu_mi - the paramater to organize the execution of mutation and migration
    %         PGLP_iter - the number of iterations in PGLP
    % output: Mod - the modularity of the detected community structure
    %         chromosome - chromosomes in the population
    %         Division - the detected community structure
    %         time - running time

    if isequal(adjacent_array, adjacent_array') == 0
        adjacent_array = adjacent_array + adjacent_array';
    end

    % set the diagonal elements of an adjacent matrix to be 0
    [row] = find(diag(adjacent_array));

    for id = row
        adjacent_array(id, id) = 0;
    end

    [edge_begin_end] = CreatEdgeList(adjacent_array);

    num_node = size(adjacent_array, 2);
    num_edge = sum(sum(adjacent_array)) / 2; % num_edge = length(find(adjacent_array))/2;
    children_proportion = 1; % the proportion of the number of child inidividuals to pop_size
    DynQ = zeros(maxgen, 1); % modularity calculated from the adjacent matrix
    whole_size = ceil((1 + children_proportion) * pop_size);

    tic;
    [chromosome] = Initial_PGLP(pop_size, adjacent_array, num_node, PGLP_iter);
    [chromosome] = Calculate_Q(chromosome, pop_size, num_node, num_edge, adjacent_array);
    [chromosome] = Sort_Q(chromosome, pop_size, 1);

    DynQ(1, 1) = chromosome(1).fitness_1;
    disp(['time_stamp=1; ', '0 : Q_genome=', num2str(DynQ(1, 1)), '; Modularity=', ...
              num2str(chromosome(1).fitness_2)]);

    for i = 1:maxgen % the i-th iteration
        % generate offspring
        selected_pop_id = [];

        for j = (pop_size + 1):2:whole_size
            % select 2 different individuals from population to cossover
            while isempty(selected_pop_id) || selected_pop_id(1) == selected_pop_id(2)
                selected_pop_id = randi(pop_size, 1, 2);
            end

            % crossover
            [chromosome(j), chromosome(j + 1)] = Crossover(chromosome, selected_pop_id, num_node);
        end

        for pop_id = (pop_size + 1):whole_size

            if rand(1) < p_mu_mi
                % mutation
                chromosome(pop_id) = Mutation(chromosome(pop_id), p_mutation, num_edge, edge_begin_end);
            else
                % migration
                chromosome(pop_id) = Migration(chromosome(pop_id), num_node, adjacent_array, p_migration);
            end

        end

        [chromosome] = Calculate_Q(chromosome, whole_size, num_node, num_edge, adjacent_array);
        [chromosome] = Sort_Q(chromosome, whole_size, 1); % sort chromosomes by Q
        chromosome(101:200) = []; % clear up chromosomes with low modularity values
        DynQ(i, 1) = chromosome(1).fitness_1;
        disp(['time_stamp=1; ', num2str(i), ': Q_genome=', num2str(DynQ(i, 1)), ...
                '; Modularity=', num2str(chromosome(1).fitness_2)]);
    end

    [Division] = chromosome(1).clusters;
    Mod = Modularity(adjacent_array, chromosome(1).clusters);
    time = toc;
end

function [Mod, chromosome, Division, time] = DECS_2(adjacent_array, maxgen, pop_size, ...
        p_mutation, p_migration, p_mu_mi, num_neighbor, pre_Result, PGLP_iter)
    % detect the community structure at the time step

    % input:  adjacent_array - the adjacent matrix
    %         maxgen - the maximum number of iterations
    %         pop_size - the population size
    %         p_mutation - the mutation rate
    %         p_migration - the migration rate
    %         p_mu_mi - the paramater to organize the execution of mutation and migration
    %         num_neighbor - the neighbor size for each subproblem in decomposition-based multi-objective optimization
    %         pre_Result - the detected community structure at the last time step
    %         PGLP_iter - the number of iterations in PGLP
    % output: Mod - the modularity of the detected community structure
    %         chromosome - chromosomes in the population
    %         Division - the detected community structure
    %         time - running time

    global idealp weights neighbors;

    if isequal(adjacent_array, adjacent_array') == 0
        adjacent_array = adjacent_array + adjacent_array';
    end

    % set the diagonal elements of an adjacent matrix to be 0
    [row] = find(diag(adjacent_array));

    for id = row
        adjacent_array(id, id) = 0;
    end

    [edge_begin_end] = CreatEdgeList(adjacent_array);
    num_node = size(adjacent_array, 2); % the number of nodes
    num_edge = sum(sum(adjacent_array)) / 2; % the number of edges

    child_chromosome = struct('genome', {}, 'clusters', {}, 'fitness_1', {}, 'fitness_2', {});

    %% Initialization based on Tchebycheff approach
    tic;
    EP = []; % non-dominated solution set
    idealp = -Inf * ones(1, 2); % the reference point (z1,z2)
    % find neighbor solutions to each subproblems
    [weights, neighbors] = init_weight(pop_size, num_neighbor);
    [chromosome] = Initial_PGLP(pop_size, adjacent_array, num_node, PGLP_iter);
    % calculate the values of modularity and NMI
    [chromosome] = evaluate_objectives(chromosome, pop_size, num_node, ...
        num_edge, adjacent_array, pre_Result);
    % find the reference point after initialization
    f = [];

    for j = 1:pop_size
        f = [f; chromosome(j).fitness_1];
        idealp = min(f);
    end

    %% Compute snapshot and temporal costs
    % the iteration process
    for t = 1:maxgen % the t-th iteration
        % implement operations on individuals selected from neighbors
        for pop_id = 1:2:pop_size
            selected_neighbor_id = [];
            selected_pop_id = [];
            % select two different individuals from neighbors
            while isempty(selected_neighbor_id) || selected_neighbor_id(1) == selected_neighbor_id(2)
                selected_neighbor_id = randi(num_neighbor, 1, 2);
            end

            selected_pop_id = neighbors(pop_id, selected_neighbor_id);
            % execute operators to reproduce the offspring
            % crossover
            [child_chromosome(pop_id), child_chromosome(pop_id + 1)] = ...
                Crossover(chromosome, selected_pop_id, num_node);

            for m = 0:1

                if rand(1) < p_mu_mi
                    % mutate
                    child_chromosome(pop_id + m) = Mutation(child_chromosome(pop_id + m), ...
                        p_mutation, num_edge, edge_begin_end);
                else
                    % migrate
                    child_chromosome(pop_id + m) = Migration(child_chromosome(pop_id + m), ...
                        num_node, adjacent_array, p_migration);
                end

                % calculate the fitness values of clustering results
                child_chromosome(pop_id + m) = evaluate_objectives(child_chromosome(pop_id + m), 1, ...
                    num_node, num_edge, adjacent_array, pre_Result);

                % update the population
                for k = neighbors(pop_id + m, :)
                    child_fit = decomposedFitness(weights(k, :), child_chromosome(pop_id + m).fitness_1, idealp);
                    gbest_fit = decomposedFitness(weights(k, :), chromosome(k).fitness_1, idealp);

                    if child_fit < gbest_fit
                        chromosome(k).genome = child_chromosome(pop_id + m).genome;
                        chromosome(k).clusters = child_chromosome(pop_id + m).clusters;
                        chromosome(k).fitness_1 = child_chromosome(pop_id + m).fitness_1;
                        chromosome(k).fitness_2 = child_chromosome(pop_id + m).fitness_2;
                    end

                end

            end

        end

        %% Find non-dominated solutions
        for pop_id = 1:pop_size
            % non-dominated sorting -- coded Q and NMI
            if isempty(EP)
                EP = [EP chromosome(pop_id)];
            else
                isDominate = 0;
                isExist = 0;
                rmindex = [];

                for k = 1:numel(EP) % numel returns the number of elements

                    if isequal(chromosome(pop_id).clusters, EP(k).clusters) % isequal(chromosome(pop_id).genome, EP(k).genome)
                        isExist = 1;
                    end

                    if dominate(chromosome(pop_id), EP(k))
                        rmindex = [rmindex k];
                    elseif dominate(EP(k), chromosome(pop_id))
                        isDominate = 1;
                    end

                end

                EP(rmindex) = [];

                if ~isDominate && ~isExist
                    EP = [EP chromosome(pop_id)];
                end

            end

            % update the reference point
            idealp = min([child_chromosome(pop_id).fitness_1; idealp]);
        end

    end

    Modularity = [];

    for front = EP
        Modularity = [Modularity; abs(front.fitness_2(1))];
    end

    [~, index] = max(Modularity);
    Division = EP(index).clusters; % restore the optimal solution, i.e., the network division with high quality
    % dynPop{timestep_num,r} = chromosome;
    Mod = -EP(index).fitness_2(1); % decoded positive "+" modularity
    time = toc;

end

function [chromosome] = Initial_PGLP(population_size, adj_mat, node_num, PGLP_iter)
    % generate the initial population by PGLP
    chromosome = struct('genome', {}, 'clusters', {}, 'fitness_1', {}, 'fitness_2', {});
    pop_X = zeros(population_size, node_num);

    for population_id = 1:population_size
        edgeslist = edges_list(adj_mat, node_num);
        pop_X(population_id, :) = PGLP(node_num, edgeslist, PGLP_iter);
        temp = adj_mat;

        for cluster_id = 1:max(pop_X(population_id, :))
            node_id = find(pop_X(population_id, :) == cluster_id); %node_id in the same cluster

            if length(node_id) > 1
                pl = nchoosek(node_id, 2);

                for i = 1:size(pl, 1)
                    temp(pl(i, 1), pl(i, 2)) = -adj_mat(pl(i, 1), pl(i, 2));
                    temp(pl(i, 2), pl(i, 1)) = -adj_mat(pl(i, 2), pl(i, 1));
                end

                % else: only one single node here, do nothing
            end

        end

        chromosome(population_id).genome =- temp;
        chromosome(population_id).clusters = 0;
        chromosome(population_id).fitness_1 = 0.00;
        chromosome(population_id).fitness_2 = 0.00;
    end

end

function [chromosome] = Migration(chromosome, node_num, adj_mat, p_migration)
    % execute migration on a chromosome

    % the nodes' communities in a vector
    [clu_assignment] = change(chromosome, 1, node_num);
    % find the nodes in each community
    clu_num = max(clu_assignment);
    index = {};

    for i = 1:clu_num % cluster_id
        index{i, 1} = find(clu_assignment == i);
    end

    for j = 1:clu_num
        num_node_in_clu = length(index{j, 1}); % the number of nodes in community j

        k = 1;

        while k <= num_node_in_clu && num_node_in_clu ~= 0
            S = adj_mat(index{j, 1}, index{j, 1});

            sum_inter = [];
            neighbor_cluster = [];
            node_id = [];
            neighbor_nodes = [];

            node_id = index{j, 1}(k);
            sum_intra = sum(S(k, :)); % the total nunmber of edges intra-connecting the node
            neighbor_nodes = find(adj_mat(node_id, :) == 1); % the neighbors in the network
            neighbor_cluster = unique(clu_assignment(1, neighbor_nodes)); % find the community of each neighbor
            neighbor_cluster(neighbor_cluster == j) = [];
            len = length(neighbor_cluster);

            if len == 0
                k = k + 1;
            else % len > 0
                sum_inter(:, 1) = neighbor_cluster'; % the community id
                sum_inter(:, 2) = zeros(len, 1); % the edges connecting the nodes in other communities

                for l = 1:len
                    % check if a node is a strongly-, neturally- or weakly-neighbor node
                    neighbor_clu_id = neighbor_cluster(l);
                    sum_inter(l, 2) = sum(adj_mat(index{neighbor_clu_id, 1}, node_id));
                end

                max_inter = max(sum_inter(:, 2));
                temp_id = find(sum_inter(:, 2) == max_inter);
                % randomly select one of candidate communities
                max_inter_id = sum_inter(temp_id(randi(length(temp_id), 1)), 1);

                %% Migration on 3 kinds of nodes
                if sum_intra < max_inter % for a weakly-neighbor node
                    % inter-connected to the nodes which is originally intra-connected
                    orgn_edge = find(chromosome.genome(node_id, :) == 1); % the original intra-connected  nodes
                    chromosome.genome(orgn_edge, node_id) = -1;
                    chromosome.genome(node_id, orgn_edge) = -1;
                    % choose a candidate community to join in
                    a = find(chromosome.genome(index{max_inter_id, 1}, node_id) == -1);
                    new_edge = index{max_inter_id, 1}(a);
                    % randomly select nodes in the selected community to be intra-connect
                    num_selected_edge = randi(length(new_edge));
                    selected_edge_sort = randperm(length(new_edge), num_selected_edge);
                    selected_edge = new_edge(selected_edge_sort);
                    chromosome.genome(selected_edge, node_id) = 1;
                    chromosome.genome(node_id, selected_edge) = 1;

                    % update nodes' communities
                    clu_assignment(1, node_id) = max_inter_id;
                    % updates the nodes in each community
                    index{j, 1}(k) = []; % remove
                    index{max_inter_id, 1} = [index{max_inter_id, 1} node_id]; % add
                    num_node_in_clu = num_node_in_clu - 1;
                end

                if sum_intra == max_inter % for a neturally-neighbor node

                    if rand(1) > p_migration % choose a candidate community to join in
                        orgn_edge = find(chromosome.genome(node_id, :) == 1); % intra-connected edges
                        chromosome.genome(orgn_edge, node_id) = -1;
                        chromosome.genome(node_id, orgn_edge) = -1;
                        a = chromosome.genome(index{max_inter_id, 1}, node_id) == -1;
                        % intra-connected to nodes in the selected cadidate community
                        new_edge = index{max_inter_id, 1}(a);
                        chromosome.genome(new_edge, node_id) = 1;
                        chromosome.genome(node_id, new_edge) = 1;
                        % update nodes' communities
                        clu_assignment(1, node_id) = max_inter_id;
                        % updates the nodes in each community
                        index{j, 1}(k) = []; % remove
                        index{max_inter_id, 1} = [index{max_inter_id, 1} node_id]; % add
                        num_node_in_clu = num_node_in_clu - 1;
                    end

                end

                if sum_intra > max_inter % for a strongly-neighbor node
                    k = k + 1;
                end

            end

        end

    end

end

function [y] = Modularity(adj_mat, clu_assignment)
    %% n: the number of clusters
    n = max(clu_assignment);

    %% L: the total number of edges in the network
    L = sum(sum(adj_mat)) / 2;

    %%
    Q = 0;

    for i = 1:n
        index = find(clu_assignment == i);
        S = adj_mat(index, index);
        li = sum(sum(S)) / 2;
        di = 0;

        for j = 1:length(index)
            di = di + sum(adj_mat(index(j), :));
        end

        Q = Q + (li - ((di) ^ 2) / (4 * L));
    end

    y = Q / L;
end

function [child_chromosome] = Mutation(child_chromosome, mutation_rate, num_edge, edge_begin_end)
    % execute mutation
    num_mutation = ceil(num_edge * mutation_rate); % the number of mutated edges

    for mutation_id = 1:num_mutation
        mutation_edge_id = randi(num_edge, 1);

        child_chromosome.genome(edge_begin_end(mutation_edge_id, 2), edge_begin_end(mutation_edge_id, 3)) = ...
            -1 * child_chromosome.genome(edge_begin_end(mutation_edge_id, 2), edge_begin_end(mutation_edge_id, 3));
        child_chromosome.genome(edge_begin_end(mutation_edge_id, 3), edge_begin_end(mutation_edge_id, 2)) = ...
            -1 * child_chromosome.genome(edge_begin_end(mutation_edge_id, 3), edge_begin_end(mutation_edge_id, 2));
    end

end

function MIhat = NMI(A, B)
    % A is the cluster we get, while B is the real partition.

    if length(A) ~= length(B)
        error('length( A ) must == length( B)');
    end

    total = length(A);
    A_ids = unique(A);
    B_ids = unique(B);
    % Mutual information
    MI = 0;

    for idA = A_ids

        for idB = B_ids
            idAOccur = find(A == idA);
            idBOccur = find(B == idB);
            idABOccur = intersect(idAOccur, idBOccur);

            px = length(idAOccur) / total;
            py = length(idBOccur) / total;
            pxy = length(idABOccur) / total;

            MI = MI + pxy * log2(pxy / (px * py) + eps); % eps : the smallest positive number

        end

    end

    % Normalized Mutual information
    Hx = 0; % Entropies

    for idA = A_ids
        idAOccurCount = length(find(A == idA));
        Hx = Hx - (idAOccurCount / total) * log2(idAOccurCount / total + eps);
    end

    Hy = 0; % Entropies

    for idB = B_ids
        idBOccurCount = length(find(B == idB));
        Hy = Hy - (idBOccurCount / total) * log2(idBOccurCount / total + eps);
    end

    MIhat = 2 * MI / (Hx + Hy);
end

function [x] = PGLP(node_num, el, iters)
    % Population Generation via Label Propagation(PGLP)
    % el = edges_list(adj_mat,node_num)

    x = 1:node_num;
    x = perturbation(x);

    for i = 1:iters
        temp_x = x;

        for j = 1:node_num
            nb_labels = temp_x(el(j).e);
            nb_size = el(j).n;

            if nb_size > 0
                t = tabulate(nb_labels);
                max_nb_labels = t(t(:, 2) == max(t(:, 2)), 1);
                x(j) = max_nb_labels(randi(length(max_nb_labels)));
            end

        end

    end

    x = sorting(x);
end

function [x] = perturbation(x)
    %��һ��ϵ�е�Ԫ��˳�����
    %%
    n = length(x);
    i = n;

    while i > 0
        index = randi(n);
        temp = x(i);
        x(i) = x(index);
        x(index) = temp;
        i = i - 1;
    end

end

function [X] = sorting(X)
    flag = 1;
    tempX = X;

    for i = 1:length(X)

        if tempX(i) ~= -1

            for j = i + 1:length(X)

                if tempX(i) == tempX(j)
                    X(j) = flag;
                    tempX(j) = -1;
                end

            end

            tempX(i) = -1;
            X(i) = flag;
            flag = flag + 1;
        end

    end

end

function [chromosome] = Sort_Q(chromosome, whole_size, signal)
    % sort the chromosomes by modularity:
    % signal = 1��sort chromosomes by the modularity value calculated from the adjacent matrix;
    % signal = 2��sort chromosomes by the modularity values calculated from the chromosome vector

    if signal == 1

        for i = 1:whole_size
            k = i;

            for j = i + 1:whole_size

                if chromosome(j).fitness_1(1) > chromosome(k).fitness_1(1)
                    k = j;
                end

            end

            if k ~= i
                temp = chromosome(i);
                chromosome(i) = chromosome(k);
                chromosome(k) = temp;
            end

        end

    end

    if signal == 2

        for i = 1:whole_size
            k = i;

            for j = i + 1:whole_size

                if chromosome(j).fitness_2 > chromosome(k).fitness_2
                    k = j;
                end

                % compare the modularity values calculated from the adjacent matrix
                % when the modularity values calculated from the chromosome vector are the same
                if chromosome(j).fitness_2(1) == chromosome(k).fitness_2(1) && chromosome(j).fitness_1(1) > chromosome(k).fitness_1(1)
                    k = j;
                end

            end

            if k ~= i
                temp = chromosome(i);
                chromosome(i) = chromosome(k);
                chromosome(k) = temp;
            end

        end

    end

end

function [node_chrom] = change(chromosome, pop_size, num_node)
    % transform the genoem matrix into the vector whose elements
    % represent the id of communities to which nodes belong

    % input:
    %       choromosome - all chromosomes in the population
    %       pop_size - the population size
    %       num_node - the number of nodes in the network
    % output:
    %       node_chrom - a row of node_chrom represents the clustering of an
    %       individual, which displays the id of communities to which nodes belong

    node_chrom = zeros(pop_size, num_node);

    for population_id = 1:pop_size
        flag = zeros(1, num_node);
        cluster_id = 1;
        node_chrom(population_id, 1) = cluster_id;

        for row_id = 1:num_node

            if flag(row_id) == 0
                flag(row_id) = 1;
                [node_chrom, flag] = row_change(chromosome(population_id).genome, node_chrom, flag, population_id, num_node, cluster_id, row_id);
                cluster_id = cluster_id + 1;
            end

        end

    end

end

function tc_fit = decomposedFitness(weight, objectives, idealpoint)
    weight((weight == 0)) = 0.00001; % wight>0
    part2 = abs(objectives - idealpoint);
    tc_fit = max(weight .* part2, [], 2);
end

function b = dominate(x, y)

    if isfield(x, 'fitness_1')
        x = x.fitness_1;
    end

    if isfield(y, 'fitness_1')
        y = y.fitness_1;
    end

    b = all(x <= y) && any(x < y);
end

function [et] = edges_list(A, V)
    %EDGELIST Generate the edges list for each node.
    et = [];

    for i = 1:V
        et(i).e = [];
        et(i).n = 0;

        for j = 1:V

            if (A(i, j) == 1)
                et(i).e = [et(i).e j];
                et(i).n = et(i).n + 1;
            end

        end

    end

end

function [chromosome] = evaluate_objectives(chromosome, whole_size, node_num, edge_num, adjacent_array, pre_cluster)
    % calculate the modularity from the genome matrix and chromosome vector, respectively
    % input:
    %       choromosome - all chromosomes in the population
    %       whole_size - the total number of the original and new individuals
    %       num_node - the number of nodes in the network
    %       num_edge - the number of edges in the network
    %       adjacent_array - the adjacent matrix of the network
    %       pre_cluster - the clustering result at the previous time step
    % output:
    %       choromosome - all chromosomes in the population

    % transform the genoem matrix into the vector whose elements
    % represent the community to which a node belongs
    [node_chrom] = change(chromosome, whole_size, node_num);

    for pop_id = 1:whole_size
        clusters_num = max(node_chrom(pop_id, :));
        e = zeros(1, clusters_num);
        a = zeros(1, clusters_num);

        for j = 1:clusters_num
            cluster_id = j;
            nodes_in_cluster = find(node_chrom(pop_id, :) == cluster_id); % find the nodes within the same community
            L = length(nodes_in_cluster); % L - the number of nodes in a community

            for k = 1:L

                for m = 1:node_num

                    if adjacent_array(nodes_in_cluster(k), m) == 1 % find the node's neighbors
                        % check if nodes are clustered into the same community
                        if chromosome(pop_id).genome(nodes_in_cluster(k), m) == 1
                            e(cluster_id) = e(cluster_id) + 1;
                        else
                            a(cluster_id) = a(cluster_id) + 1;
                        end

                    end

                end

            end

        end

        e = e ./ 2;
        a = a ./ 2;
        a = a + e;
        e = e / edge_num;
        a = (a / edge_num) .^ 2;
        Q = 0;

        for n = 1:clusters_num
            Q = Q + e(n) - a(n);
        end

        chromosome(pop_id).fitness_1(1) =- Q; % (-) modularity calculated from the genome matrix
        chromosome(pop_id).fitness_1(2) =- NMI(node_chrom(pop_id, :), pre_cluster); % temporal smoothness
        chromosome(pop_id).clusters = node_chrom(pop_id, :); % the clustering result
        chromosome(pop_id).fitness_2(1) =- Modularity(adjacent_array, chromosome(pop_id).clusters); % modularity
    end

    function [weights, neighbors] = init_weight(popsize, niche)
        % init_weights and neighbors.
        weights = [];

        for i = 0:popsize - 1
            weight = zeros(1, 2);
            weight(1) = i / (popsize - 1);
            weight(2) = (popsize - i - 1) / (popsize - 1);

            % weight(1) = i/popsize;
            % weight(2) = (popsize-i)/popsize;
            weights = [weights; weight];
        end

        % Set up the neighborhood
        leng = size(weights, 1);
        distanceMatrix = zeros(leng, leng);

        for i = 1:leng

            for j = i + 1:leng
                A = weights(i, :)'; B = weights(j, :)';
                distanceMatrix(i, j) = (A - B)' * (A - B);
                distanceMatrix(j, i) = distanceMatrix(i, j);
            end

            [~, sindex] = sort(distanceMatrix(i, :));
            neighbors(i, :) = sindex(1:niche);
        end

    end

    function [node_chrom, flag] = row_change(genome, node_chrom, flag, ...
            population_id, node_num, cluster_id, row_id)
        node_chrom(population_id, row_id) = cluster_id;

        for colum_id = 1:node_num

            if genome(row_id, colum_id) == 1 && flag(colum_id) == 0
                flag(colum_id) = 1;
                [node_chrom, flag] = row_change(genome, node_chrom, flag, population_id, node_num, cluster_id, colum_id);
            end

        end

        clear all;
        clc;

        %% Load a dataset
        flag = 2; % set flag = 1 for synthetic networks or flag = 2 for real-world networks
        % synthetic networks
        % load('datasets/syn_fix_3.mat');
        % load('datasets/syn_fix_5.mat');
        % load('datasets/syn_var_3.mat');
        % load('datasets/syn_var_5.mat');
        % load('datasets/expand.mat');
        % load('datasets/mergesplit.mat');

        % real-world networks
        % the gound truth community structures are returned by the first step of DYNMOGA
        % load('datasets/cell.mat');
        % load('datasets/firststep_DYNMOGA_cell.mat');
        load('datasets/enron.mat');
        load('datasets/firststep_DYNMOGA_enron.mat');
        GT_Cube = dynMoeaResult;

        %% Parameter setting
        maxgen = 100; % the maximum number of iterations
        pop_size = 100; % the population size
        num_neighbor = 5; % the neighbor size for each subproblem in decomposition-based multi-objective optimization
        p_mutation = 0.20; % the mutation rate
        p_migration = 0.50; % the migration rate
        p_mu_mi = 0.50; % the paramater to organize the execution of mutation and migration
        PGLP_iter = 5; % the number of iterations in PGLP
        num_repeat = 5; % the number of repeated run

        %% Results at each time step
        dynMod = []; % modularity of detected community structure
        dynNmi = []; % NMI between detected community structure and the ground truth
        dynPop = {}; % the population
        dynTime = []; % the running time
        DECS_Result = {}; % the detected community structure

        for r = 1:num_repeat
            %     global idealp weights neighbors;
            % idealp is reference point (z1, z2) where z1 and z2
            % are the maximum of the 1st and 2nd objective functions
            num_timestep = size(W_Cube, 2); % W_Cube contains several cells restoring temporal adjacent matrices
            %% DECS only optimizes the modularity at the 1st time step
            timestep_num = 1;
            [dynMod(1, r), dynPop{1, r}, DECS_Result{1, r}, dynTime(1, r)] = ...
                DECS_1(W_Cube{timestep_num}, maxgen, pop_size, p_mutation, p_migration, p_mu_mi, PGLP_iter);
            % calculate NMI for synthetic or real-world networks
            if flag == 1
                % for synthetic networks
                dynNmi(1, r) = NMI(GT_Matrix(:, 1)', DECS_Result{1, r});
            else
                % for real-world networks
                dynNmi(1, r) = NMI(GT_Cube{timestep_num}, DECS_Result{1, r});
            end

            disp(['timestep = ', num2str(timestep_num), ', Modularity = ', ...
                      num2str(dynMod(timestep_num, r)), ', NMI = ', num2str(dynNmi(timestep_num, r))]);

            %% DECS optimizes the modularity and NMI in the following time steps
            for timestep_num = 2:num_timestep
                [dynMod(timestep_num, r), dynPop{timestep_num, r}, DECS_Result{timestep_num, r}, ...
                     dynTime(timestep_num, r)] = DECS_2(W_Cube{timestep_num}, maxgen, pop_size, ...
                    p_mutation, p_migration, p_mu_mi, num_neighbor, DECS_Result{timestep_num - 1, r}, PGLP_iter);

                if flag == 1
                    dynNmi(timestep_num, r) = NMI(DECS_Result{timestep_num, r}, GT_Matrix(:, timestep_num)');
                else
                    dynNmi(timestep_num, r) = NMI(DECS_Result{timestep_num, r}, GT_Cube{timestep_num});
                end

                disp(['timestep = ', num2str(timestep_num), ', Modularity = ', ...
                          num2str(dynMod(timestep_num, r)), ', NMI = ', num2str(dynNmi(timestep_num, r))]);
            end

        end

        avg_dynMod = sum(dynMod, 2) / num_repeat;
        avg_dynNmi = sum(dynNmi, 2) / num_repeat;
        avg_dynMod = sum(dynMod, 2) / num_repeat;
