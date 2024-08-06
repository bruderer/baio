class AniseedAPI:
    BASE_URL = "http://www.aniseed.fr/api"

    def all_genes(self, organism_id, search=None):
        """
        Returns a URL to list all genes for a given organism.
        Optionally, a search term can be provided to filter the genes.
        """
        url = f"{self.BASE_URL}/all_genes?organism_id={organism_id}"
        if search:
            url += f"&search={search}"
        return url

    def all_genes_by_stage(self, organism_id, stage, search=None):
        """
        Returns a URL to list all genes for a given organism that are expressed at a
        specific stage.
        Optionally, a search term can be provided to filter the genes.
        """
        url = (
            f"{self.BASE_URL}/all_genes_by_stage?"
            f"organism_id={organism_id}&stage={stage}"
        )
        if search:
            url += f"&search={search}"
        return url

    def all_genes_by_stage_range(
        self, organism_id, start_stage, end_stage, search=None
    ):
        """
        Returns a URL to list all genes for a given organism that are expressed between
        two stages.
        Optionally, a search term can be provided to filter the genes.
        """
        url = (
            f"{self.BASE_URL}/all_genes_by_stage_range?"
            f"organism_id={organism_id}&start_stage={start_stage}&end_stage={end_stage}"
        )
        if search:
            url += f"&search={search}"
        return url

    def all_genes_by_territory(self, organism_id, cell, search=None):
        """
        Returns a URL to list all genes for a given organism that are expressed in a
        specific territory. Optionally, a search term can be provided to filter the
        genes.
        """
        url = (
            f"{self.BASE_URL}/all_genes_by_territory?"
            f"organism_id={organism_id}&cell={cell}"
        )
        if search:
            url += f"&search={search}"
        return url

    def all_territories_by_gene(self, organism_id, gene, search=None):
        """
        Returns a URL to list all territories where a specific gene is expressed for a
        given organism.
        Optionally, a search term can be provided to filter the territories.
        """
        url = (
            f"{self.BASE_URL}/all_territories_by_gene?"
            f"organism_id={organism_id}&gene={gene}"
        )
        if search:
            url += f"&search={search}"
        return url

    def all_clones_by_gene(self, organism_id, gene, search=None):
        """
        Returns a URL to list all clones for a specific gene for a given organism.
        Optionally, a search term can be provided to filter the clones.
        """
        url = f"{self.BASE_URL}/clones?organism_id={organism_id}&gene={gene}"
        if search:
            url += f"&search={search}"
        return url

    def all_constructs(self, organism_id, search=None):
        """
        Returns a URL to list all constructs for a given organism.
        Optionally, a search term can be provided to filter the constructs.
        """
        url = f"{self.BASE_URL}/constructs?organism_id={organism_id}"
        if search:
            url += f"&search={search}"
        return url

    def all_molecular_tools(self, search=None):
        """
        Returns a URL to list all molecular tools in the database.
        Optionally, a search term can be provided to filter the tools.
        """
        url = f"{self.BASE_URL}/molecular_tools"
        if search:
            url += f"?search={search}"
        return url

    def all_publications(self, search=None):
        """
        Returns a URL to list all publications in the database.
        Optionally, a search term can be provided to filter the publications.
        """
        url = f"{self.BASE_URL}/publications"
        if search:
            url += f"?search={search}"
        return url

    def all_regulatory_regions(self, organism_id, search=None):
        """
        Returns a URL to list all regulatory regions for a given organism.
        Optionally, a search term can be provided to filter the regions.
        """
        url = f"{self.BASE_URL}/regulatory_regions?organism_id={organism_id}"
        if search:
            url += f"&search={search}"
        return url


def execute_query(api, query):
    # Create a dictionary mapping parameter names to their values
    # print('in execute_query')
    params = {
        query.parameter_1_name: query.parameter_1_value,
        query.parameter_2_name: query.parameter_2_value,
        query.parameter_3_name: query.parameter_3_value,
    }
    # print(f'Params first: {params}')
    # Remove any parameters that were not filled or are None
    params = {k: v for k, v in params.items() if v not in [None, "To be filled", ""]}
    print(params)
    # print(f'Params second: {params}')

    # Get the required function from the API
    func = getattr(api, query.required_function)
    print(func)
    # print(f'Func: {func}')

    # Call the function with the parameters
    return func(**params)
